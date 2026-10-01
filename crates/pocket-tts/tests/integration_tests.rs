//! Integration tests for TTSModel with real weights
//!
//! These tests require the HF_TOKEN environment variable to be set
//! for downloading model weights from HuggingFace.

use pocket_tts::TTSModel;
use pocket_tts::audio::{read_wav, write_wav};
use pocket_tts::voice_state::init_states;
use pocket_tts::weights::download_if_necessary;

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Static OnceLock to share the model across tests, preventing concurrent downloads.
static MODEL: OnceLock<TTSModel> = OnceLock::new();

/// Static OnceLock for model loaded with custom params (for pause tests).
static MODEL_WITH_PARAMS: OnceLock<TTSModel> = OnceLock::new();
/// Shared lock for all gated model/token operations to avoid HF cache lock races.
static MODEL_INIT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// HF_TOKEN, or a token saved by `hf auth login`.
fn has_hf_token() -> bool {
    let env = std::env::var("HF_TOKEN").is_ok_and(|v| !v.trim().is_empty());
    let cached = std::env::var_os("HOME")
        .map(|h| std::path::Path::new(&h).join(".cache/huggingface/token"))
        .is_some_and(|p| p.is_file());
    env || cached
}

fn require_hf_token(test_name: &str) -> bool {
    if has_hf_token() {
        true
    } else {
        eprintln!("Skipping {test_name}: no Hugging Face token (HF_TOKEN or hf auth login)");
        false
    }
}

fn model_init_lock() -> &'static Mutex<()> {
    MODEL_INIT_LOCK.get_or_init(|| Mutex::new(()))
}

/// Get or initialize the shared TTSModel instance.
/// This ensures the model is loaded exactly once, preventing lock contention
/// when multiple tests run in parallel.
fn get_model() -> &'static TTSModel {
    let _guard = model_init_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    MODEL.get_or_init(|| TTSModel::load("english").expect("Failed to load model"))
}

/// Get or initialize the shared TTSModel instance with custom parameters.
fn get_model_with_params() -> &'static TTSModel {
    let _guard = model_init_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    MODEL_WITH_PARAMS.get_or_init(|| {
        TTSModel::load_with_params(
            "english",
            0.0,
            pocket_tts::config::defaults::LSD_DECODE_STEPS,
            pocket_tts::config::defaults::EOS_THRESHOLD,
        )
        .expect("Failed to load model with params")
    })
}

fn get_ref_wav_path() -> PathBuf {
    // pocket-tts -> crates -> project_root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets")
        .join("ref.wav")
}

#[test]
fn test_download_non_gated_tokenizer() {
    // Test downloading from non-gated repo (pocket-tts-without-voice-cloning)
    let path = "hf://kyutai/pocket-tts-without-voice-cloning/tokenizer.model@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3";
    let result = download_if_necessary(path);
    assert!(result.is_ok(), "Failed to download: {:?}", result.err());
    let local_path = result.unwrap();
    assert!(
        local_path.exists(),
        "Downloaded file does not exist: {:?}",
        local_path
    );
    println!("Downloaded to: {:?}", local_path);
}

#[test]
// #[ignore = "requires HF_TOKEN and gated model access"]
fn test_download_gated_weights() {
    if !require_hf_token("test_download_gated_weights") {
        return;
    }

    // Test downloading from gated repo (pocket-tts)
    let _guard = model_init_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let path =
        "hf://kyutai/pocket-tts/tts_b6369a24.safetensors@427e3d61b276ed69fdd03de0d185fa8a8d97fc5b";
    let result = download_if_necessary(path);
    assert!(result.is_ok(), "Failed to download: {:?}", result.err());
}

#[test]
// #[ignore = "requires HF_TOKEN and model download"]
fn test_tts_model_load() {
    if !require_hf_token("test_tts_model_load") {
        return;
    }
    let model = get_model();
    assert_eq!(model.sample_rate, 24000);
    assert_eq!(model.dim, 1024);
    assert_eq!(model.ldim, 32);
}

/// covers: REQ-VOI-002
#[test]
// #[ignore = "requires HF_TOKEN and model download"]
fn test_voice_cloning_from_ref_wav() {
    if !require_hf_token("test_voice_cloning_from_ref_wav") {
        return;
    }
    let model = get_model();

    let ref_wav_path = get_ref_wav_path();
    if !ref_wav_path.exists() {
        eprintln!("ref.wav not found at {:?}, skipping test", ref_wav_path);
        return;
    }

    let voice_state = model
        .get_voice_state(&ref_wav_path)
        .expect("Failed to get voice state");

    // Voice state should have entries from running through the transformer
    assert!(!voice_state.is_empty(), "Voice state should not be empty");
}

/// covers: REQ-INF-002
#[test]
// #[ignore = "requires HF_TOKEN and model download"]
fn test_audio_generation_produces_valid_output() {
    if !require_hf_token("test_audio_generation_produces_valid_output") {
        return;
    }
    let model = get_model();

    let ref_wav_path = get_ref_wav_path();
    if !ref_wav_path.exists() {
        eprintln!("ref.wav not found at {:?}, skipping test", ref_wav_path);
        return;
    }

    let voice_state = model
        .get_voice_state(&ref_wav_path)
        .expect("Failed to get voice state");

    let audio = model
        .generate("Hello world.", &voice_state)
        .expect("Failed to generate audio");

    // Check output shape
    let dims = audio.dims();
    assert_eq!(dims.len(), 2, "Audio should be [channels, samples]");
    assert_eq!(dims[0], 1, "Should have 1 channel");
    assert!(dims[1] > 0, "Should have some samples");

    // Audio should be reasonable length (at least 0.1 seconds for "Hello world")
    let duration_seconds = dims[1] as f32 / model.sample_rate as f32;
    assert!(
        duration_seconds > 0.1,
        "Audio should be at least 0.1 seconds, got {}",
        duration_seconds
    );

    // Optional: save to file for manual inspection
    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_output.wav");
    write_wav(&output_path, &audio, model.sample_rate as u32).expect("Failed to write audio");
    println!("Test audio written to {:?}", output_path);
}

#[test]
// #[ignore = "requires HF_TOKEN and model download"]
fn test_mimi_encode_decode_roundtrip() {
    if !require_hf_token("test_mimi_encode_decode_roundtrip") {
        return;
    }
    let model = get_model();

    let ref_wav_path = get_ref_wav_path();
    if !ref_wav_path.exists() {
        eprintln!("ref.wav not found at {:?}, skipping test", ref_wav_path);
        return;
    }

    let (audio, sample_rate) = read_wav(&ref_wav_path).expect("Failed to read ref.wav");

    // Resample if needed
    let audio = if sample_rate != model.sample_rate as u32 {
        pocket_tts::audio::resample(&audio, sample_rate, model.sample_rate as u32)
            .expect("Failed to resample")
    } else {
        audio
    };

    // Add batch dimension: [C, T] -> [B, C, T]
    let audio = audio.unsqueeze(0).expect("Failed to add batch dim");

    // Pad audio to a multiple of frame size
    let frame_size = model.mimi.frame_size();
    let (b, c, t) = audio.dims3().expect("dims3");
    let pad_len = if t % frame_size != 0 {
        frame_size - (t % frame_size)
    } else {
        0
    };
    let audio = if pad_len > 0 {
        let pad =
            candle_core::Tensor::zeros((b, c, pad_len), audio.dtype(), audio.device()).unwrap();
        candle_core::Tensor::cat(&[&audio, &pad], 2).unwrap()
    } else {
        audio
    };

    // Encode
    let mut encode_state = init_states(1, 1000);
    let latent = model
        .mimi
        .encode_to_latent(&audio, &mut encode_state, 0)
        .expect("Failed to encode");

    println!("Encoded latent shape: {:?}", latent.dims());

    // Decode (the quantizer projection maps the 32-channel latent back to 512)
    let latent = model
        .mimi
        .quantize(&latent)
        .expect("Failed to project latent");
    let mut decode_state = init_states(1, 1000);
    let decoded = model
        .mimi
        .decode_from_latent(&latent, &mut decode_state, 0)
        .expect("Failed to decode");

    println!("Decoded audio shape: {:?}", decoded.dims());

    // The decoded audio should have similar length (within a frame)
    let original_len = audio.dims()[2];
    let decoded_len = decoded.dims()[2];

    // Allow for some length difference due to framing
    let max_diff = 1920 * 2; // ~2 frames at 24kHz
    let len_diff = (original_len as i64 - decoded_len as i64).unsigned_abs() as usize;
    assert!(
        len_diff < max_diff,
        "Audio length mismatch: original={}, decoded={}, diff={}",
        original_len,
        decoded_len,
        len_diff
    );
}

/// covers: REQ-TXT-002
#[test]
// #[ignore = "requires HF_TOKEN and model download"]
fn test_generate_with_pauses_adds_silence() {
    if !require_hf_token("test_generate_with_pauses_adds_silence") {
        return;
    }
    let model = get_model_with_params();

    let ref_wav_path = get_ref_wav_path();
    if !ref_wav_path.exists() {
        eprintln!("ref.wav not found at {:?}, skipping test", ref_wav_path);
        return;
    }

    let voice_state = model
        .get_voice_state(&ref_wav_path)
        .expect("Failed to get voice state");

    // Generate with pause (should be longer due to silence)
    let text_with_pause = "Hello [pause:500ms] world.";
    let audio_with_pause = model
        .generate_with_pauses(text_with_pause, &voice_state)
        .expect("Failed to generate audio with pauses");

    // "Hello" and "world." are generated separately, so the total length is
    // not "Hello world." plus 500 ms; check the silence itself instead.
    let samples: Vec<f32> = audio_with_pause.flatten_all().unwrap().to_vec1().unwrap();
    let mut longest = 0;
    let mut run = 0;
    for x in &samples {
        run = if *x == 0.0 { run + 1 } else { 0 };
        longest = longest.max(run);
    }
    assert!(
        longest >= 12_000,
        "expected a 500 ms (12000-sample) run of silence, longest is {longest}"
    );
    assert!(
        samples.len() > 12_000 + 2 * 1920,
        "only {} samples",
        samples.len()
    );
}

// Tests that don't require model download
#[test]
fn test_pause_module_integration() {
    use pocket_tts::parse_text_with_pauses;

    let parsed = parse_text_with_pauses("Hello [pause:500ms] world... [pause:1s] done");

    // Should have clean text without pause markers
    assert!(!parsed.clean_text.contains("[pause:"));

    // Should have detected pauses
    assert!(parsed.pauses.len() >= 2, "Should have at least 2 pauses");

    // Check pause durations
    let has_500ms = parsed.pauses.iter().any(|p| p.duration_ms == 500);
    let has_1000ms = parsed.pauses.iter().any(|p| p.duration_ms == 1000);
    assert!(has_500ms, "Should have 500ms pause");
    assert!(has_1000ms, "Should have 1000ms (1s) pause");
}
