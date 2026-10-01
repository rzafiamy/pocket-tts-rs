//! `pocket-tts convert`: pack a model into one GGUF file.

use anyhow::{Context, Result};
use clap::Parser;
use owo_colors::OwoColorize;
use pocket_tts::weights::download_if_necessary;
use std::path::PathBuf;

#[derive(Parser, Debug)]
pub struct ConvertArgs {
    /// Model variant to convert (`english`, `french`, `french_24l`, ...) or a
    /// config YAML path
    #[arg(long, default_value = "english")]
    pub variant: String,

    /// Dtype of the linear layers: f32, f16, q8_0, q6k, q5k, q4k, q4_0
    #[arg(long, default_value = "q8_0")]
    pub dtype: String,

    /// Comma-separated predefined voices to embed ("all" for every voice;
    /// default: the language's default voice)
    #[arg(long)]
    pub voices: Option<String>,

    /// Output file (default: <variant>-<dtype>.gguf)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

pub fn run(args: ConvertArgs) -> Result<()> {
    let dtype = pocket_tts::gguf::parse_dtype(&args.dtype)?;
    let config_yaml = match pocket_tts::builtin_configs::get(&args.variant) {
        Some(y) => y.to_string(),
        None => std::fs::read_to_string(&args.variant)
            .with_context(|| format!("unknown variant or config path '{}'", args.variant))?,
    };
    let config = pocket_tts::config::parse_config(&config_yaml)?;
    let variant = std::path::Path::new(&args.variant)
        .file_stem()
        .map_or(args.variant.clone(), |s| s.to_string_lossy().into_owned());

    println!("{} Fetching {} weights...", "▶".cyan(), variant.yellow());
    let weights_path = config
        .weights_path
        .as_deref()
        .context("config has no weights_path")?;
    let (weights, cloning) = match download_if_necessary(weights_path) {
        Ok(w) => (w, true),
        Err(e) => {
            let fallback = config
                .weights_path_without_voice_cloning
                .as_deref()
                .ok_or(e)?;
            println!(
                "  {} voice-cloning weights unavailable (gated repo, see README); using weights without voice cloning",
                "!".yellow()
            );
            (download_if_necessary(fallback)?, false)
        }
    };
    let tokenizer = download_if_necessary(&config.flow_lm.lookup_table.tokenizer_path)?;

    let voice_names: Vec<String> = match args.voices.as_deref() {
        Some("all") => pocket_tts::voices::PREDEFINED_VOICES
            .iter()
            .map(|v| v.to_string())
            .collect(),
        Some("") | Some("none") => Vec::new(),
        Some(list) => list.split(',').map(|v| v.trim().to_string()).collect(),
        None => vec![pocket_tts::voices::default_voice(&variant).to_string()],
    };
    let mut voices = Vec::new();
    for name in voice_names {
        if !pocket_tts::voices::is_predefined(&variant, &name) {
            anyhow::bail!("'{name}' is not a predefined voice of {variant}");
        }
        let url = pocket_tts::voices::predefined_voice_url(&variant, &name);
        voices.push((name, download_if_necessary(&url)?));
    }

    let output = args
        .output
        .unwrap_or_else(|| PathBuf::from(format!("{variant}-{}.gguf", args.dtype.to_lowercase())));
    println!("{} Writing {}...", "▶".cyan(), output.display());
    let counts = pocket_tts::gguf::convert(
        &pocket_tts::gguf::ConvertOptions {
            variant: &variant,
            config_yaml: &config_yaml,
            weights: &weights,
            tokenizer: &tokenizer,
            dtype,
            voices,
        },
        &output,
    )?;
    let size = std::fs::metadata(&output)?.len() as f64 / 1e6;
    let summary: Vec<String> = counts.iter().map(|(d, n)| format!("{n} {d}")).collect();
    println!(
        "  {} {} ({size:.1} MB; tensors: {}; voice cloning: {})",
        "✓".green(),
        output.display(),
        summary.join(", "),
        if cloning { "yes" } else { "no" }
    );
    Ok(())
}
