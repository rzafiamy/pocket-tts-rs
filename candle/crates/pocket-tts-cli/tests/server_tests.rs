use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use pocket_tts::TTSModel;
use pocket_tts_cli::server::{routes, state::AppState};
use serde_json::json;
use tower::ServiceExt; // for oneshot

// Helper to create app with mock or real model?
// Creating a real model is heavy (downloads weights).
// We might want to mock the model, but TTSModel is a struct, not a trait.
// For integration tests, we might skip the heavy weight loading if we can,
// but for "End-to-End" verification we need it.
// Given this is the "Phase 5 verification", we *should* load the real model if possible,
// but it might be too slow for CI.
// However, the prompt implies using `ref.wav` etc, so let's try to load it.
// To avoid re-loading per test, we might use `lazy_static` or `OnceCell` but that's complex for tests.
// Let's just create one test that does multiple things to amortize load time.

#[tokio::test]
async fn test_api_full_flow() {
    // Only run if model weights exist, otherwise skip (to avoid massive download in CI environment if not cached)
    // Actually, `TTSModel::load` will try to download.
    // Let's assume the environment is set up (since user has the repo).

    println!("Loading model for API test...");
    let model = match TTSModel::load("b6369a24") {
        Ok(m) => m,
        Err(e) => {
            println!("Skipping API test: could not load model: {}", e);
            return;
        }
    };

    let state = AppState::new(model);
    let app = routes::create_router(state);

    // 1. Health Check
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // 2. Generate Audio
    // Use a very short text to be fast
    let body = json!({
        "text": "Hi",
        // uses default voice
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

    // 3. OpenAI endpoint
    let body = json!({
        "model": "pocket-tts",
        "input": "Open API test",
        "voice": "alba"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers().get("content-type").unwrap(), "audio/wav");
}
