use anyhow::Result;
use clap::Parser;
use pocket_tts::TTSModel;
use std::path::PathBuf;

#[derive(Parser, Debug)]
pub struct GenerateArgs {
    /// Text to synthesize
    #[arg(short, long)]
    pub text: String,

    /// Path to voice reference audio for voice cloning (optional)
    /// Also accepts predefined voice names like 'alba', 'marius', etc.
    #[arg(short, long)]
    pub voice: Option<String>,

    /// Output file path
    #[arg(short, long, default_value = "output.wav")]
    pub output: PathBuf,

    /// Sampling temperature (default: 0.7)
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// LSD decode steps (default: 1)
    #[arg(long, default_value = "1")]
    pub lsd_decode_steps: usize,

    /// EOS threshold (default: -4.0)
    #[arg(long, default_value = "-4.0")]
    pub eos_threshold: f32,

    /// Stream audio to stdout (raw 16-bit PCM)
    #[arg(long)]
    pub stream: bool,
}

pub fn run(args: GenerateArgs) -> Result<()> {
    if !args.stream {
        println!("Loading model...");
    }
    let model = TTSModel::load_with_params(
        "b6369a24",
        args.temperature,
        args.lsd_decode_steps,
        args.eos_threshold,
    )?;

    // Predefined voices
    let predefined_voices = [
        "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
    ];

    let voice_state = if let Some(ref v) = args.voice {
        if predefined_voices.contains(&v.as_str()) {
            if !args.stream {
                println!("Using predefined stock voice: {}", v);
            }
            // Try to find in HF cache on D:
            let cache_path = format!(
                "D:\\huggingface\\hub\\models--kyutai--pocket-tts-without-voice-cloning\\snapshots\\d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3\\embeddings\\{}.safetensors",
                v
            );
            let path = std::path::PathBuf::from(cache_path);
            if path.exists() {
                model.get_voice_state_from_prompt_file(path)?
            } else {
                anyhow::bail!(
                    "Predefined voice '{}' found in config but .safetensors file not found at {:?}",
                    v,
                    path
                );
            }
        } else {
            if !args.stream {
                println!("Using voice cloning from: {:?}", v);
            }
            model.get_voice_state(v)?
        }
    } else {
        // Default to alba stock voice
        if !args.stream {
            println!("No voice specified, defaulting to stock voice: alba");
        }
        let cache_path = "D:\\huggingface\\hub\\models--kyutai--pocket-tts-without-voice-cloning\\snapshots\\d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3\\embeddings\\alba.safetensors";
        let path = std::path::PathBuf::from(cache_path);
        if path.exists() {
            model.get_voice_state_from_prompt_file(path)?
        } else {
            if !args.stream {
                println!(
                    "Warning: alba.safetensors not found in cache, using empty state (not recommended)"
                );
            }
            pocket_tts::voice_state::init_states(1, 1000)
        }
    };

    if args.stream {
        use std::io::Write;
        let mut stdout = std::io::stdout();

        // Generate stream
        for chunk_res in model.generate_stream(&args.text, &voice_state) {
            let chunk = chunk_res?;
            // Convert tensor to Vec<u8> (16-bit PCM)
            let chunk = chunk.squeeze(0)?;
            let data = chunk.to_vec2::<f32>()?;
            for (i, _) in data[0].iter().enumerate() {
                for channel_data in &data {
                    let val = (channel_data[i].clamp(-1.0, 1.0) * 32767.0) as i16;
                    stdout.write_all(&val.to_le_bytes())?;
                }
            }
            stdout.flush()?;
        }
    } else {
        use candle_core::Tensor;
        use indicatif::{ProgressBar, ProgressStyle};

        println!("Generating: \"{}\"", args.text);

        let total_steps = model.estimate_generation_steps(&args.text) as u64;
        let pb = ProgressBar::new(total_steps);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("##-"),
        );
        pb.set_message("Generating...");

        let mut audio_chunks = Vec::new();
        let mut total_samples = 0;

        for chunk_res in model.generate_stream(&args.text, &voice_state) {
            let chunk = chunk_res?;
            let dims = chunk.dims();
            let samples = if dims.len() == 2 { dims[1] } else { dims[0] };
            total_samples += samples;

            audio_chunks.push(chunk);
            pb.inc(1); // Increment by 1 step (1 chunk)
            pb.set_message(format!(
                "{:.2}s",
                total_samples as f32 / model.sample_rate as f32
            ));
        }

        pb.finish_with_message("Done");

        // Concatenate all audio chunks
        if audio_chunks.is_empty() {
            anyhow::bail!("No audio generated");
        }
        let audio = Tensor::cat(&audio_chunks, 2)?;
        // Remove batch dimension
        let audio = audio.squeeze(0)?;

        let dims = audio.dims();
        println!("Audio shape: {:?}", dims);

        let num_samples = if dims.len() == 2 { dims[1] } else { dims[0] };
        let duration_sec = num_samples as f32 / model.sample_rate as f32;

        println!("Saving to: {:?}", args.output);
        pocket_tts::audio::write_wav(&args.output, &audio, model.sample_rate as u32)?;

        println!(
            "Done! Generated {} samples ({:.2}s at {}Hz)",
            num_samples, duration_sec, model.sample_rate
        );
    }

    Ok(())
}
