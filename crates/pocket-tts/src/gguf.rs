//! Single-file GGUF models: weights (linear layers optionally quantized),
//! config, tokenizer and predefined voices.
//!
//! Layout:
//! - metadata `general.architecture = "pocket-tts"`, `pocket_tts.variant`,
//!   `pocket_tts.config` (YAML), `pocket_tts.tokenizer` (tokenizer.json),
//!   `pocket_tts.voices` (names of the embedded voices);
//! - every safetensors tensor under its original name. 2-D `*.weight`
//!   matrices of linear layers use the requested dtype; everything else
//!   (convolutions, norms, the text embedding table) stays F32;
//! - voice `<name>` as `voice.<name>.<module>.cache`, the attention cache
//!   `[2, T, H, D]` trimmed to the prompt length, in F16.

use crate::ModelState;
use crate::config::Config;
use crate::modules::linear::QuantWeights;
use anyhow::{Context, Result};
use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub const ARCH: &str = "pocket-tts";

/// Linear weights small enough to keep dense (the EOS head is one row).
const KEEP_DENSE: &[&str] = &["flow_lm.out_eos.weight"];

/// Parses a dtype name: f32, f16, q8_0, q6k, q5k, q4k, q4_0.
pub fn parse_dtype(name: &str) -> Result<GgmlDType> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "f32" => GgmlDType::F32,
        "f16" => GgmlDType::F16,
        "bf16" => GgmlDType::BF16,
        "q8_0" | "q8" => GgmlDType::Q8_0,
        "q6k" | "q6_k" => GgmlDType::Q6K,
        "q5k" | "q5_k" => GgmlDType::Q5K,
        "q4k" | "q4_k" => GgmlDType::Q4K,
        "q4_0" => GgmlDType::Q4_0,
        other => anyhow::bail!("unknown dtype '{other}' (f32, f16, q8_0, q6k, q5k, q4k, q4_0)"),
    })
}

fn is_linear_weight(name: &str, t: &Tensor) -> bool {
    t.rank() == 2
        && name.ends_with(".weight")
        && !name.contains("conditioner.embed")
        && !KEEP_DENSE.contains(&name)
}

/// Dtype for a linear weight: `wanted` if the row length fits its block,
/// else the closest one that does.
fn linear_dtype(wanted: GgmlDType, row: usize) -> GgmlDType {
    for dt in [wanted, GgmlDType::Q8_0, GgmlDType::F16] {
        if row.is_multiple_of(dt.block_size()) {
            return dt;
        }
    }
    GgmlDType::F32
}

/// What to put in a GGUF file.
pub struct ConvertOptions<'a> {
    pub variant: &'a str,
    pub config_yaml: &'a str,
    pub weights: &'a Path,
    pub tokenizer: &'a Path,
    pub dtype: GgmlDType,
    /// (name, exported model-state .safetensors) for each embedded voice.
    pub voices: Vec<(String, std::path::PathBuf)>,
}

/// Writes a GGUF model; returns per-dtype tensor counts for reporting.
pub fn convert(opts: &ConvertOptions, out: &Path) -> Result<Vec<(String, usize)>> {
    let cpu = Device::Cpu;
    let tensors = candle_core::safetensors::load(opts.weights, &cpu)?;
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    let mut qtensors: Vec<(String, QTensor)> = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for name in names {
        let t = tensors[name].to_dtype(DType::F32)?;
        let dt = if is_linear_weight(name, &t) {
            linear_dtype(opts.dtype, t.dim(1)?)
        } else {
            GgmlDType::F32
        };
        *counts.entry(format!("{dt:?}")).or_default() += 1;
        qtensors.push((name.clone(), QTensor::quantize(&t, dt)?));
    }

    let mut voice_names = Vec::new();
    for (voice, path) in &opts.voices {
        let state = candle_core::safetensors::load(path, &cpu)?;
        for (key, cache) in &state {
            let Some(module) = key.strip_suffix("/cache") else {
                continue;
            };
            let offset = state_offset(&state, module)?;
            // [2, B=1, T, H, D] -> [2, offset, H, D]
            let cache = cache
                .to_dtype(DType::F32)?
                .narrow(2, 0, offset)?
                .squeeze(1)?
                .contiguous()?;
            qtensors.push((
                format!("voice.{voice}.{module}.cache"),
                QTensor::quantize(&cache, GgmlDType::F16)?,
            ));
        }
        voice_names.push(gguf_file::Value::String(voice.clone()));
    }

    let tokenizer = std::fs::read_to_string(opts.tokenizer).with_context(|| {
        format!(
            "reading {:?} (only tokenizer.json is supported)",
            opts.tokenizer
        )
    })?;
    let metadata = [
        (
            "general.architecture",
            gguf_file::Value::String(ARCH.into()),
        ),
        (
            "pocket_tts.variant",
            gguf_file::Value::String(opts.variant.into()),
        ),
        (
            "pocket_tts.config",
            gguf_file::Value::String(opts.config_yaml.into()),
        ),
        ("pocket_tts.tokenizer", gguf_file::Value::String(tokenizer)),
        ("pocket_tts.voices", gguf_file::Value::Array(voice_names)),
    ];
    let metadata: Vec<(&str, &gguf_file::Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> =
        qtensors.iter().map(|(n, t)| (n.as_str(), t)).collect();

    let mut file = std::io::BufWriter::new(std::fs::File::create(out)?);
    gguf_file::write(&mut file, &metadata, &tensor_refs)?;

    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort();
    Ok(counts)
}

fn state_offset(state: &HashMap<String, Tensor>, module: &str) -> Result<usize> {
    if let Some(o) = state.get(&format!("{module}/offset")) {
        return Ok(o.flatten_all()?.to_dtype(DType::I64)?.to_vec1::<i64>()?[0] as usize);
    }
    if let Some(end) = state.get(&format!("{module}/current_end")) {
        return Ok(end.dim(0)?);
    }
    anyhow::bail!("{module}: no offset in exported state")
}

/// Everything read from a GGUF model file.
pub struct GgufModel {
    pub variant: String,
    pub config: Config,
    pub tokenizer_json: String,
    /// Dense tensors (F32 on `device`), for the regular `VarBuilder`.
    pub dense: HashMap<String, Tensor>,
    /// Quantized linear weights, consumed through `QMatMul`.
    pub quant: Arc<QuantWeights>,
    /// Embedded voices, as FlowLM attention states.
    pub voices: HashMap<String, ModelState>,
}

fn meta_str(content: &gguf_file::Content, key: &str) -> Result<String> {
    match content.metadata.get(key) {
        Some(gguf_file::Value::String(s)) => Ok(s.clone()),
        _ => anyhow::bail!("GGUF metadata '{key}' missing or not a string"),
    }
}

pub fn read(path: &Path, device: &Device) -> Result<GgufModel> {
    let mut file = std::fs::File::open(path).with_context(|| format!("opening {path:?}"))?;
    let content = gguf_file::Content::read(&mut file)?;
    let arch = meta_str(&content, "general.architecture")?;
    if arch != ARCH {
        anyhow::bail!("{path:?} is a '{arch}' GGUF, not a {ARCH} model");
    }
    let variant = meta_str(&content, "pocket_tts.variant")?;
    let config: Config = serde_yaml::from_str(&meta_str(&content, "pocket_tts.config")?)?;
    let tokenizer_json = meta_str(&content, "pocket_tts.tokenizer")?;

    let mut dense = HashMap::new();
    let mut quant = QuantWeights::new();
    let mut voice_tensors: HashMap<String, Vec<(String, Tensor)>> = HashMap::new();
    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    for name in names {
        let qt = content.tensor(&mut file, &name, device)?;
        if let Some(rest) = name.strip_prefix("voice.") {
            // voice.<name>.<module>.cache
            let (voice, module) = rest.split_once('.').context("bad voice tensor name")?;
            let module = module
                .strip_suffix(".cache")
                .context("bad voice tensor name")?;
            voice_tensors
                .entry(voice.to_string())
                .or_default()
                .push((module.to_string(), qt.dequantize(device)?));
        } else if matches!(qt.dtype(), GgmlDType::F32) {
            dense.insert(name, qt.dequantize(device)?);
        } else {
            quant.insert(name, Arc::new(qt));
        }
    }

    let mut voices = HashMap::new();
    for (voice, modules) in voice_tensors {
        let mut state = ModelState::new();
        for (module, cache) in modules {
            // [2, T, H, D] -> k, v as [1, H, T, D]
            let t = cache.dim(1)?;
            let k = cache.get(0)?.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
            let v = cache.get(1)?.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
            state.insert(
                format!("flow_lm.{module}"),
                crate::voice_state::attention_state(k, v, t, device)?,
            );
        }
        voices.insert(voice, state);
    }

    Ok(GgufModel {
        variant,
        config,
        tokenizer_json,
        dense,
        quant: Arc::new(quant),
        voices,
    })
}
