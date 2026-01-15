use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use base64::{Engine as _, engine::general_purpose};
use pocket_tts::TTSModel;
use pocket_tts_cli::server::{routes, state::AppState};
use serde_json::json;
use std::path::Path;
use tower::ServiceExt;

#[tokio::test]
async fn test_api_base64_cloning() {
    println!("Loading model for Base64 API test...");
    // Assume we can load model - reuse logic from server_tests (though this is a separate file potentially)
    // For simplicity, let's copy the setup.
    let model = match TTSModel::load("b6369a24") {
        Ok(m) => m,
        Err(e) => {
            println!("Skipping API test: could not load model: {}", e);
            return;
        }
    };

    let state = AppState::new(model);
    let app = routes::create_router(state);

    // Read ref.wav to bytes and base64 encode
    let ref_wav = "../../ref.wav";
    if !Path::new(ref_wav).exists() {
        println!("Skipping base64 test: ref.wav not found");
        return;
    }

    let wav_bytes = std::fs::read(ref_wav).unwrap();
    let b64 = general_purpose::STANDARD.encode(&wav_bytes);

    let body = json!({
        "text": "Base64 cloning test",
        "voice": b64
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers().get("content-type").unwrap(), "audio/wav");

    // Check if body is valid wav
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let cursor = std::io::Cursor::new(bytes);
    let reader = hound::WavReader::new(cursor).unwrap();
    assert!(reader.duration() > 0);
}
