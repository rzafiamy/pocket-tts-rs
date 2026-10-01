//! GGUF conversion and loading on the real French model. Skipped when the
//! weights cannot be downloaded (no network, or no Hugging Face access).

use pocket_tts::TTSModel;
use pocket_tts::gguf::{ConvertOptions, convert, parse_dtype};
use pocket_tts::weights::download_if_necessary;
use std::path::PathBuf;

const VARIANT: &str = "french";
const TEXT: &str = "Bonjour le monde. Ceci est un test.";

struct Files {
    weights: PathBuf,
    tokenizer: PathBuf,
    voice: PathBuf,
}

fn fetch() -> Option<Files> {
    let config = pocket_tts::tts_model::resolve_config(VARIANT).ok()?;
    let weights = config
        .weights_path
        .as_deref()
        .and_then(|p| download_if_necessary(p).ok())
        .or_else(|| {
            config
                .weights_path_without_voice_cloning
                .as_deref()
                .and_then(|p| download_if_necessary(p).ok())
        });
    let Some(weights) = weights else {
        eprintln!("skipping: cannot download the {VARIANT} weights");
        return None;
    };
    let tokenizer = download_if_necessary(&config.flow_lm.lookup_table.tokenizer_path).ok()?;
    let voice = download_if_necessary(&pocket_tts::voices::predefined_voice_url(
        VARIANT, "estelle",
    ))
    .ok()?;
    Some(Files {
        weights,
        tokenizer,
        voice,
    })
}

fn write(files: &Files, dtype: &str) -> PathBuf {
    let out = std::env::temp_dir().join(format!(
        "pocket-tts-test-{}-{dtype}.gguf",
        std::process::id()
    ));
    convert(
        &ConvertOptions {
            variant: VARIANT,
            config_yaml: pocket_tts::builtin_configs::get(VARIANT).unwrap(),
            weights: &files.weights,
            tokenizer: &files.tokenizer,
            dtype: parse_dtype(dtype).unwrap(),
            voices: vec![("estelle".into(), files.voice.clone())],
        },
        &out,
    )
    .expect("convert");
    out
}

fn speak(model: &mut TTSModel) -> Vec<f32> {
    model.temp = 0.0; // zero noise: deterministic
    let voice = model
        .embedded_voices
        .get("estelle")
        .cloned()
        .unwrap_or_else(|| {
            model
                .get_voice_state_from_prompt_file(&fetch().unwrap().voice)
                .unwrap()
        });
    model
        .generate(TEXT, &voice)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&x| x as f64).sum::<f64>() / n,
        b.iter().map(|&x| x as f64).sum::<f64>() / n,
    );
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        sab += dx * dy;
        saa += dx * dx;
        sbb += dy * dy;
    }
    sab / (saa * sbb).sqrt()
}

/// covers: REQ-GGF-001, REQ-GGF-002
/// An f32 GGUF reproduces the safetensors model (same length, corr > 0.999)
/// and carries its config, tokenizer and voice.
#[test]
fn f32_gguf_matches_safetensors() {
    let Some(files) = fetch() else { return };
    let path = write(&files, "f32");
    let mut from_gguf = TTSModel::load_gguf(&path, &candle_core::Device::Cpu).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(from_gguf.variant, VARIANT);
    assert!(from_gguf.embedded_voices.contains_key("estelle"));

    let mut reference = TTSModel::load(VARIANT).unwrap();
    let a = speak(&mut from_gguf);
    let b = speak(&mut reference);
    assert_eq!(a.len(), b.len(), "different lengths");
    // Not bit-exact: threaded reductions sum in varying order and the
    // autoregressive loop amplifies the last bits.
    assert!(correlation(&a, &b) > 0.999, "corr {}", correlation(&a, &b));
}

/// covers: REQ-GGF-003
/// A q8_0 GGUF stays under 180 MB (f32: 442 MB) and still produces 1-8 s
/// of non-silent audio for two short sentences.
#[test]
fn q8_gguf_is_small_and_speaks() {
    let Some(files) = fetch() else { return };
    let q8 = write(&files, "q8_0");
    let size = std::fs::metadata(&q8).unwrap().len();
    let mut model = TTSModel::load_gguf(&q8, &candle_core::Device::Cpu).unwrap();
    std::fs::remove_file(&q8).ok();
    assert!(size < 180_000_000, "q8_0 file is {size} bytes");

    let audio = speak(&mut model);
    let seconds = audio.len() as f32 / model.sample_rate as f32;
    assert!((1.0..8.0).contains(&seconds), "{seconds} s of audio");
    let peak = audio.iter().map(|x| x.abs()).fold(0f32, f32::max);
    assert!(peak > 0.05, "silent output (peak {peak})");
}
