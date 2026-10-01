//! Pocket TTS CLI - Rust/Candle port
//!
//! A blazingly fast text-to-speech tool.

use anyhow::Result;
use clap::Parser;

use pocket_tts_cli::commands;

/// Pocket TTS - High-quality text-to-speech, blazingly fast on CPU
#[derive(Parser)]
#[command(
    name = "pocket-tts",
    author,
    version,
    about = "Pocket TTS - Blazingly fast text-to-speech",
    long_about = "A Rust/Candle port of Kyutai's Pocket TTS model.\n\n\
                  Generate natural speech from text using neural TTS.\n\
                  Supports voice cloning from audio samples."
)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Generate audio from text
    ///
    /// Synthesizes speech from the provided text and saves to a WAV file.
    /// Supports voice cloning using predefined voices or custom audio files.
    Generate(commands::generate::GenerateArgs),

    /// Start the HTTP API server
    ///
    /// Runs a web server providing TTS generation via REST API,
    /// including an OpenAI-compatible /v1/audio/speech endpoint.
    Serve(commands::serve::ServeArgs),

    /// Convert a model to a single GGUF file
    ///
    /// Packs weights (linear layers quantized to --dtype), config, tokenizer
    /// and predefined voices into one file for `--model`.
    Convert(commands::convert::ConvertArgs),

    /// Load a model and print its effective configuration
    Info(commands::info::InfoArgs),
}

fn main() -> Result<()> {
    let args = Args::parse();
    let threads = match &args.command {
        Commands::Generate(a) => a.model.threads,
        Commands::Serve(a) => a.model.threads,
        Commands::Info(a) => a.model.threads,
        Commands::Convert(_) => None,
    };
    // Before the async runtime starts its worker threads.
    pocket_tts_cli::loader::configure_threads(threads);

    match args.command {
        Commands::Generate(cmd_args) => {
            // Generate is CPU-bound, run synchronously
            commands::generate::run(cmd_args)
        }
        Commands::Serve(cmd_args) => {
            tokio::runtime::Runtime::new()?.block_on(commands::serve::run(cmd_args))
        }
        Commands::Convert(cmd_args) => commands::convert::run(cmd_args),
        Commands::Info(cmd_args) => commands::info::run(cmd_args),
    }
}
