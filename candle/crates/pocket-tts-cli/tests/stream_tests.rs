use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use pocket_tts::TTSModel;
use pocket_tts_cli::server::{routes, state::AppState};
use serde_json::json;
use tokio_stream::StreamExt;
use tower::ServiceExt; // for collecting stream

#[tokio::test]
async fn test_api_stream_endpoint() {
    println!("Loading model for Stream API test...");
    let model = match TTSModel::load("b6369a24") {
        Ok(m) => m,
        Err(e) => {
            println!("Skipping API test: could not load model: {}", e);
            return;
        }
    };

    let state = AppState::new(model);
    let app = routes::create_router(state);

    let ref_wav = std::path::PathBuf::from("d:\\pocket-tts-candle\\ref.wav");

    let body = json!({
        "text": "Streaming test",
        "voice": ref_wav.to_str().unwrap()
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/stream")
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "application/octet-stream"
    );

    // Collect stream
    let mut stream = response.into_body().into_data_stream();
    let mut total_bytes = 0;
    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res.expect("Stream chunk error");
        total_bytes += chunk.len();
    }

    println!("Total streamed bytes: {}", total_bytes);
    assert!(total_bytes > 0);
}
