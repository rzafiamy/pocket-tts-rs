//! Times model load, voice load and generation; prints the real-time factor.
//! Usage: cargo run --release --example bench -- <variant> [voice] [runs] [cuda]
use candle_core::Device;
use pocket_tts::TTSModel;
use std::time::Instant;

const TEXT_EN: &str = "The history of artificial intelligence is a fascinating journey. \
    Early systems focused on symbolic reasoning, while modern methods learn from data. \
    Today, small models can even speak on a laptop.";
const TEXT_FR: &str = "L'histoire de l'intelligence artificielle est un voyage fascinant. \
    Les premiers systèmes reposaient sur le raisonnement symbolique, les méthodes modernes \
    apprennent à partir des données. Aujourd'hui, de petits modèles parlent même sur un portable.";

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let variant = args.get(1).map(String::as_str).unwrap_or("english");
    let runs: usize = args.get(3).and_then(|r| r.parse().ok()).unwrap_or(3);
    let device = if args.get(4).is_some_and(|d| d == "cuda") {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };

    let t = Instant::now();
    let model = load(variant, &device)?;
    let variant = model.variant.as_str();
    println!("load: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let text = if variant.starts_with("french") {
        TEXT_FR
    } else {
        TEXT_EN
    };

    let voice = args
        .get(2)
        .filter(|v| !v.is_empty())
        .cloned()
        .unwrap_or_else(|| pocket_tts::voices::default_voice(variant).to_string());
    let t = Instant::now();
    let url = pocket_tts::voices::predefined_voice_url(variant, &voice);
    let state = if let Some(s) = model.embedded_voices.get(&voice) {
        s.clone()
    } else if pocket_tts::voices::is_predefined(variant, &voice) {
        model.get_voice_state_from_prompt_file(pocket_tts::weights::download_if_necessary(&url)?)?
    } else {
        model.get_voice_state(&voice)?
    };
    println!("voice: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);

    for run in 0..runs {
        let t = Instant::now();
        let mut first_chunk_ms = None;
        let mut samples = 0;
        for chunk in model.generate_stream(text, &state) {
            let chunk = chunk?;
            if first_chunk_ms.is_none() {
                first_chunk_ms = Some(t.elapsed().as_secs_f64() * 1e3);
            }
            samples += chunk.dim(2)?;
        }
        let elapsed = t.elapsed().as_secs_f64();
        let audio = samples as f64 / model.sample_rate as f64;
        println!(
            "run {run}: {audio:.2} s audio in {elapsed:.2} s -> {:.2}x real-time, first chunk {:.0} ms",
            audio / elapsed,
            first_chunk_ms.unwrap_or(0.0)
        );
    }
    Ok(())
}

/// A built-in variant name or a `.gguf` file.
fn load(spec: &str, device: &Device) -> anyhow::Result<TTSModel> {
    if spec.ends_with(".gguf") {
        return TTSModel::load_gguf(spec, device);
    }
    let mut m = TTSModel::load_with_params_device(spec, 0.3, 1, -4.0, None, device)?;
    m.variant = spec.to_string();
    Ok(m)
}
