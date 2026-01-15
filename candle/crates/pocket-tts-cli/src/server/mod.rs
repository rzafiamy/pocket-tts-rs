use anyhow::Result;
use pocket_tts::TTSModel;

pub mod handlers;
pub mod routes;
pub mod state;

pub async fn start_server(host: &str, port: u16) -> Result<()> {
    // Initialize tracing if not already initialized (might fail if called twice, but for CLI it's fine)
    let _ = tracing_subscriber::fmt::try_init();

    println!("Loading model...");
    // Load model synchronously before starting server
    let model = TTSModel::load("b6369a24")?;

    let state = state::AppState::new(model);
    let app = routes::create_router(state);

    let addr = format!("{}:{}", host, port);
    println!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
