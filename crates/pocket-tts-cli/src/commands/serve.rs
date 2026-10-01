//! Serve command implementation
//!
//! Provides `pocket-tts serve` for HTTP API server.

use anyhow::Result;
use clap::{ArgAction, Parser};
use owo_colors::OwoColorize;

use crate::voice::PREDEFINED_VOICES;

#[derive(Parser, Debug, Clone)]
pub struct ServeArgs {
    /// Host address to bind (default: 127.0.0.1)
    #[arg(long, env = "POCKET_TTS_HOST", default_value = "127.0.0.1")]
    pub host: String,

    /// Port number to listen on (default: 8000)
    #[arg(short, long, env = "POCKET_TTS_PORT", default_value_t = 8000)]
    pub port: u16,

    /// Default voice for API requests (can be overridden per-request)
    #[arg(long, env = "POCKET_TTS_VOICE")]
    pub voice: Option<String>,

    #[command(flatten)]
    pub model: crate::loader::ModelArgs,

    /// Maximum number of resolved voice states to keep in server LRU cache.
    #[arg(long, default_value_t = 64)]
    pub voice_cache_capacity: usize,

    /// Comma-separated extra voices to prewarm at startup (e.g. "alba,marius");
    /// the default voice is always loaded.
    #[arg(long, default_value = "")]
    pub prewarm_voices: String,

    /// Run a tiny startup warmup generation to reduce first-request latency.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub warmup: bool,
}

pub async fn run(args: ServeArgs) -> Result<()> {
    print_banner();

    println!(
        "{} Loading model: {}",
        "->".cyan(),
        args.model.describe().yellow()
    );

    let server_args = args.clone();
    crate::server::start_server(server_args).await
}

fn print_banner() {
    println!();
    println!(
        "  {}  {} {}",
        "[]".bold(),
        "Pocket TTS".bold().cyan(),
        "API Server".bold()
    );
    println!(
        "      {} {}",
        "Rust/Candle port".dimmed(),
        format!("v{}", env!("CARGO_PKG_VERSION")).dimmed()
    );
    println!();
}

/// Print endpoint information after server starts
pub fn print_endpoints(host: &str, port: u16) {
    let base = format!("http://{}:{}", host, port);

    println!();
    println!(
        "  {} {}",
        "[ok]".green().bold(),
        "Server is running!".green().bold()
    );
    println!();
    println!("  {}", "Endpoints:".bold());
    println!(
        "    {} {}  {}",
        "GET".cyan(),
        format!("{}/health", base).white(),
        "Health check".dimmed()
    );
    println!(
        "    {} {}  {}",
        "POST".yellow(),
        format!("{}/generate", base).white(),
        "Generate audio (JSON body)".dimmed()
    );
    println!(
        "    {} {}  {}",
        "POST".yellow(),
        format!("{}/stream", base).white(),
        "Streaming generation".dimmed()
    );
    println!(
        "    {} {}  {}",
        "POST".yellow(),
        format!("{}/tts", base).white(),
        "Python-compatible endpoint (form data)".dimmed()
    );
    println!(
        "    {} {}  {}",
        "POST".yellow(),
        format!("{}/v1/audio/speech", base).white(),
        "OpenAI-compatible".dimmed()
    );
    println!();
    println!(
        "  {} Available voices: {}",
        "Voices:".dimmed(),
        PREDEFINED_VOICES.join(", ").dimmed()
    );
    println!();
    println!(
        "  {} curl -X POST {}/generate -H 'Content-Type: application/json' -d '{{\"text\": \"Hello world\"}}' --output test.wav",
        "Example:".dimmed(),
        base
    );
    println!();
}
