use crate::server::state::AppState;
use axum::{
    Json,
    extract::{State},
    http::{StatusCode, HeaderMap},
    response::{IntoResponse, Response},
    body::Body,
};
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt as _; // For map

#[derive(Serialize)]
pub struct HealthResponse {
    status: String,
}

pub async fn health_check() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

#[derive(Deserialize)]
pub struct GenerateRequest {
    text: String,
    voice: Option<String>,
}

pub async fn generate(
    State(state): State<AppState>,
    Json(payload): Json<GenerateRequest>,
) -> Response {
    // Acquire lock to ensure sequential processing
    let _guard = state.lock.lock().await;

    let model = state.model.clone();
    let text = payload.text.clone();
    let voice = payload.voice.clone();

    // Offload to blocking thread
    let result = tokio::task::spawn_blocking(move || {
        // Resolve voice state
        // TODO: Refactor this shared logic with commands/generate.rs
        let voice_state = if let Some(ref v) = voice {
            // 1. Check for Base64 (starts with "base64:" or just looks like it if reasonable length and not a path)
            // We use a prefix heuristic "base64:" or check if it fails path check AND looks like base64?
            // Safer to use a prefix if possible, but user might send raw base64. 
            // Let's assume if it is NOT a file path and NOT a preset, we try to decode as base64.
            
            let predefined_voices = [
                 "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
             ];

             if predefined_voices.contains(&v.as_str()) {
                 let cache_path = format!(
                     "D:\\huggingface\\hub\\models--kyutai--pocket-tts-without-voice-cloning\\snapshots\\d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3\\embeddings\\{}.safetensors",
                     v
                 );
                 let path = std::path::PathBuf::from(cache_path);
                 if path.exists() {
                     model.get_voice_state_from_prompt_file(path).map_err(|e| e.to_string())?
                 } else {
                      return Err(format!("Predefined voice '{}' not found at {:?}", v, path));
                 }
             } else {
                 // Try path
                 let path = std::path::PathBuf::from(v);
                 if path.exists() {
                     model.get_voice_state(path).map_err(|e| e.to_string())?
                 } else {
                     // Try Base64
                     // Strip "data:audio/wav;base64," if present
                     let b64_str = if v.starts_with("data:") {
                         v.split(",").nth(1).unwrap_or(v)
                     } else {
                         v
                     };
                     
                     use base64::{Engine as _, engine::general_purpose};
                     match general_purpose::STANDARD.decode(b64_str) {
                         Ok(bytes) => {
                             // Decode WAV from bytes
                             let (audio, sample_rate) = pocket_tts::audio::read_wav_from_bytes(&bytes).map_err(|e| format!("Bad wav bytes: {}", e))?;
                             // Resample if necessary (dup logic from get_voice_state)
                             let audio = if sample_rate != model.sample_rate as u32 {
                                 pocket_tts::audio::resample_linear(&audio, sample_rate, model.sample_rate as u32).map_err(|e| e.to_string())?
                             } else {
                                 audio
                             };
                             // Add batch dim
                             let audio = audio.unsqueeze(0).map_err(|e| e.to_string())?;
                             model.get_voice_state_from_tensor(&audio).map_err(|e| e.to_string())?
                         },
                         Err(_) => {
                              return Err(format!("Voice not found as file/preset and failed base64 decode: {:?}", path));
                         }
                     }
                 }
             }
        } else {
             // Default voice logic
            let cache_path = "D:\\huggingface\\hub\\models--kyutai--pocket-tts-without-voice-cloning\\snapshots\\d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3\\embeddings\\alba.safetensors";
            let path = std::path::PathBuf::from(cache_path);
             if path.exists() {
                model.get_voice_state_from_prompt_file(path).map_err(|e| e.to_string())?
             } else {
                // Emergency fallback
                pocket_tts::voice_state::init_states(1, 1000)
             }
        };

        let audio = model.generate(&text, &voice_state).map_err(|e| e.to_string())?;
        
        let mut buffer = std::io::Cursor::new(Vec::new());
        pocket_tts::audio::write_wav_to_writer(&mut buffer, &audio, model.sample_rate as u32).map_err(|e| e.to_string())?;
        
        Ok::<Vec<u8>, String>(buffer.into_inner())
    }).await;

    match result {
        Ok(Ok(wav_bytes)) => {
            let mut headers = HeaderMap::new();
            headers.insert("Content-Type", "audio/wav".parse().unwrap());
            (StatusCode::OK, headers, Body::from(wav_bytes)).into_response()
        },
        Ok(Err(e)) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))).into_response()
        },
        Err(_e) => {
             (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Task join error"}))).into_response()
        }
    }
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct OpenAIRequest {
    model: String,
    input: String,
    voice: Option<String>,
    response_format: Option<String>, // 'wav', 'mp3', etc. (we only support wav)
}

pub async fn openai_speech(
    state: State<AppState>,
    Json(payload): Json<OpenAIRequest>,
) -> Response {
    // Adapter to generate
    let req = GenerateRequest {
        text: payload.input,
        voice: payload.voice,
    };
    generate(state, Json(req)).await
}

pub async fn generate_stream(
    State(state): State<AppState>,
    Json(payload): Json<GenerateRequest>,
) -> Response {
    // Acquire lock to ensure sequential processing
    // NOTE: For streaming, we need to hold the lock for the entire stream duration?
    // If we drop the lock here, another request might intervene and mess up the stateful model?
    // The model is stateful. `generate_stream` updates state step by step.
    // If we have concurrent requests, we MUST lock.
    // However, if we return a stream, the handler returns immediately.
    // We need to move the lock permit into the stream or hold it until stream ends.
    // Arc<Mutex<()>> doesn't support "moving permit into stream" easily unless we use an async mutex guard that is Send.
    // tokio::sync::MutexGuard IS Send. So we can move it into the stream logic.

    let model = state.model.clone();
    let text = payload.text.clone();
    let voice = payload.voice.clone();
    let lock = state.lock.clone();

    // Channel for sending chunks
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, anyhow::Error>>(10);

    // Spawn blocking task to generate
    tokio::spawn(async move {
        // Acquire lock asynchronously for this task. 
        // We do it inside spawn to avoid blocking the handler if lock is busy.
        // But this means the client connects and waits for lock.
        let _guard = lock.lock().await;

        // Clone tx for the blocking task
        let tx_inner = tx.clone();
        
        let result = tokio::task::spawn_blocking(move || {
            // Resolve voice state (dup logic again, sorry)
             let voice_state = if let Some(ref v) = voice {
                let predefined_voices = [
                     "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
                 ];
                 if predefined_voices.contains(&v.as_str()) {
                     let cache_path = format!(
                         "D:\\huggingface\\hub\\models--kyutai--pocket-tts-without-voice-cloning\\snapshots\\d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3\\embeddings\\{}.safetensors",
                         v
                     );
                     let path = std::path::PathBuf::from(cache_path);
                     if path.exists() {
                         model.get_voice_state_from_prompt_file(path).map_err(|e| e.to_string())?
                     } else {
                          return Err(format!("Predefined voice '{}' not found at {:?}", v, path));
                     }
                 } else {
                     let path = std::path::PathBuf::from(v);
                     if path.exists() {
                         model.get_voice_state(path).map_err(|e| e.to_string())?
                     } else {
                         // Base64 logic
                         let b64_str = if v.starts_with("data:") {
                             v.split(",").nth(1).unwrap_or(v)
                         } else {
                             v
                         };
                         use base64::{Engine as _, engine::general_purpose};
                         match general_purpose::STANDARD.decode(b64_str) {
                             Ok(bytes) => {
                                 let (audio, sample_rate) = pocket_tts::audio::read_wav_from_bytes(&bytes).map_err(|e| format!("Bad wav bytes: {}", e))?;
                                 let audio = if sample_rate != model.sample_rate as u32 {
                                     pocket_tts::audio::resample_linear(&audio, sample_rate, model.sample_rate as u32).map_err(|e| e.to_string())?
                                 } else {
                                     audio
                                 };
                                 let audio = audio.unsqueeze(0).map_err(|e| e.to_string())?;
                                 model.get_voice_state_from_tensor(&audio).map_err(|e| e.to_string())?
                             },
                             Err(_) => return Err(format!("Voice not found/decode failed: {:?}", path))
                         }
                     }
                 }
            } else {
                // Default
                let cache_path = "D:\\huggingface\\hub\\models--kyutai--pocket-tts-without-voice-cloning\\snapshots\\d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3\\embeddings\\alba.safetensors";
                let path = std::path::PathBuf::from(cache_path);
                 if path.exists() {
                    model.get_voice_state_from_prompt_file(path).map_err(|e| e.to_string())?
                 } else {
                    pocket_tts::voice_state::init_states(1, 1000)
                 }
            };
            
            // Generate stream
            // Since generate_stream returns an iterator, we iterate it here
            for chunk_res in model.generate_stream(&text, &voice_state) {
                match chunk_res {
                    Ok(chunk) => {
                        // chunk is Tensor [1, C, T] or [C, T]?
                        // generate_stream returns [C, T] usually for single item?
                        // checked code: generate_stream calls decode_from_latent which likely returns [channels, len].
                        // We need to convert it to bytes (PCM or WAV fragment).
                        // Writing a full WAV header for a stream is tricky. usually streaming APIs return raw PCM or specialized container.
                        // For simplicity, let's return raw PCM (f32 or 16-bit).
                        // Or we can just dump 16-bit PCM bytes.
                        // Users need to know sample rate (24000Hz).
                        
                        // Convert tensor to Vec<u8> (16-bit PCM)
                        let chunk = chunk.squeeze(0).map_err(|e| e.to_string())?;
                        let data = chunk.to_vec2::<f32>().map_err(|e| e.to_string())?;
                        let mut bytes = Vec::new();
                        for (i, _) in data[0].iter().enumerate() {
                            for channel_data in &data {
                                let val = (channel_data[i].clamp(-1.0, 1.0) * 32767.0) as i16;
                                bytes.extend_from_slice(&val.to_le_bytes());
                            }
                        }
                        
                        // Blocking send to async channel?
                        // We can't use async emit in blocking task easily.
                        // We use blocking send on the handle if we had one, but tx is async.
                        // We need blocking_send.
                        if let Err(_) = tx_inner.blocking_send(Ok(bytes)) {
                            break; // Receiver dropped
                        }
                    },
                    Err(e) => {
                         let _ = tx_inner.blocking_send(Err(anyhow::anyhow!(e)));
                         break;
                    }
                }
            }
            Ok::<(), String>(())
        });
        
        match result.await {
            Ok(Ok(())) => {},
            Ok(Err(text_err)) => {
                 let _ = tx.send(Err(anyhow::anyhow!("Generation error: {}", text_err))).await;
            },
            Err(join_err) => {
                 let _ = tx.send(Err(anyhow::anyhow!("Task join error: {}", join_err))).await;
            }
        }
        // _guard dropped here, releasing lock
    });

    // Convert Receiver to Stream
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    
    // Convert to axum Body stream
    let body_stream = stream.map(|res| {
        match res {
             Ok(bytes) => Ok(axum::body::Bytes::from(bytes)),
             Err(e) => Err(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())), // How to signal error in stream?
        }
    });

    Response::builder()
        .header("Content-Type", "application/octet-stream")
        .body(Body::from_stream(body_stream))
        .unwrap()
}
