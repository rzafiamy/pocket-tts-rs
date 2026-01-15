use crate::server::handlers;
use crate::server::state::AppState;
use axum::{
    Router,
    routing::{get, post},
};
use tower_http::trace::TraceLayer;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(handlers::health_check))
        .route("/generate", post(handlers::generate))
        .route("/stream", post(handlers::generate_stream))
        .route("/v1/audio/speech", post(handlers::openai_speech))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
