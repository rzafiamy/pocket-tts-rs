//! `pocket-tts info`: load a model and print its effective configuration.

use anyhow::Result;
use clap::Parser;

use crate::loader::ModelArgs;

#[derive(Parser, Debug)]
pub struct InfoArgs {
    #[command(flatten)]
    pub model: ModelArgs,
}

pub fn run(args: InfoArgs) -> Result<()> {
    let t = std::time::Instant::now();
    let model = args.model.load()?;
    let c = &model.config;
    println!("model        {}", args.model.describe());
    println!("variant      {}", model.variant);
    println!("device       {:?}", model.device);
    println!(
        "threads      {}",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "all".into())
    );
    println!(
        "backbone     {} layers, d_model {}, sampler head {}",
        c.flow_lm.transformer.num_layers, c.flow_lm.transformer.d_model, c.flow_lm.flow.flow_type
    );
    println!("temperature  {}", model.temp);
    let mut voices: Vec<&String> = model.embedded_voices.keys().collect();
    voices.sort();
    println!(
        "voices       default {}; embedded: {}",
        pocket_tts::voices::default_voice(&model.variant),
        if voices.is_empty() {
            "none (downloaded on use)".to_string()
        } else {
            voices
                .iter()
                .map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    println!("loaded in    {} ms", t.elapsed().as_millis());
    Ok(())
}
