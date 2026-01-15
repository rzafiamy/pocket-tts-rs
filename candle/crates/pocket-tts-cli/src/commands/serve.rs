use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
pub struct ServeArgs {
    /// Host to bind (default: 127.0.0.1)
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port to listen on (default: 8000)
    #[arg(long, default_value_t = 8000)]
    pub port: u16,
}

pub async fn run(args: ServeArgs) -> Result<()> {
    crate::server::start_server(&args.host, args.port).await
}
