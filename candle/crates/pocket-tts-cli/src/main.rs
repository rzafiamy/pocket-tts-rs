use anyhow::Result;
use clap::Parser;

use pocket_tts_cli::commands;

#[derive(Parser, Debug)]
#[command(author, version, about = "Pocket TTS - Rust/Candle Port")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Generate audio from text
    Generate(commands::generate::GenerateArgs),

    /// Start API server
    Serve(commands::serve::ServeArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Generate(cmd_args) => {
            // Generate is CPU bound and synchronous, run it directly
            // (or spawn_blocking if we wanted to stay async, but strict sync is fine for CLI)
            commands::generate::run(cmd_args)
        }
        Commands::Serve(cmd_args) => commands::serve::run(cmd_args).await,
    }
}
