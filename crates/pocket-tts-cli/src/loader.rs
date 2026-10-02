//! Model selection and loading options shared by `generate` and `serve`.

use anyhow::Result;
use candle_core::Device;
use clap::Args;
use pocket_tts::TTSModel;
use std::path::PathBuf;

/// Default CPU threads. Batch-1 decoding is a stream of small ops: 1 to 4
/// threads perform alike, 16+ lose up to 2x to synchronization.
const DEFAULT_THREADS: usize = 4;

#[derive(Args, Debug, Clone)]
pub struct ModelArgs {
    /// Model variant: a language (`english`, `french`, `german`, `italian`,
    /// `spanish`, `portuguese`, `dutch`), a `_24l` variant, `b6369a24`, or a
    /// config YAML path. Ignored with --model.
    #[arg(long, env = "POCKET_TTS_VARIANT", default_value = "english")]
    pub variant: String,

    /// GGUF model file written by `pocket-tts convert`
    #[arg(short, long, env = "POCKET_TTS_MODEL")]
    pub model: Option<PathBuf>,

    /// Device: cpu, cuda, cuda:N or metal
    #[arg(long, env = "POCKET_TTS_DEVICE", default_value = "cpu")]
    pub device: String,

    /// CPU threads (default: min(4, available cores))
    #[arg(long, env = "POCKET_TTS_THREADS")]
    pub threads: Option<usize>,

    /// Sampling temperature (higher = more variation; defaults to the
    /// model's recommended value)
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Sampler decode steps (more steps = better quality, slower)
    #[arg(long, default_value_t = 1)]
    pub lsd_decode_steps: usize,

    /// EOS threshold (more negative = longer audio)
    #[arg(long, default_value_t = -4.0, allow_hyphen_values = true)]
    pub eos_threshold: f32,

    /// Clamp sampling noise to [-x, x]
    #[arg(long)]
    pub noise_clamp: Option<f32>,

    /// Do not spell out numbers, times, amounts and units or strip Markdown
    /// before synthesis (French and English rules; on by default)
    #[arg(long, env = "POCKET_TTS_NO_NORMALIZE")]
    pub no_normalize: bool,

    /// Shorten the silence between generated chunks (320 ms after a sentence,
    /// 160 ms after a comma split) instead of the model's full end-of-utterance
    /// pause, which makes long sentences sound choppy; `false` keeps upstream's
    #[arg(long, env = "POCKET_TTS_TIGHT_PAUSES", default_value_t = true, action = clap::ArgAction::Set)]
    pub tight_pauses: bool,
}

/// Sets the CPU thread count. Call before any tensor work.
pub fn configure_threads(threads: Option<usize>) {
    if threads.is_none() && std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return;
    }
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    let n = threads.unwrap_or(DEFAULT_THREADS.min(available)).max(1);
    // SAFETY: called from main before any other thread is spawned.
    unsafe { std::env::set_var("RAYON_NUM_THREADS", n.to_string()) };
}

pub fn parse_device(spec: &str) -> Result<Device> {
    let (kind, index) = match spec.split_once(':') {
        Some((k, i)) => (k, i.parse()?),
        None => (spec, 0),
    };
    Ok(match kind {
        "cpu" => Device::Cpu,
        "cuda" => Device::new_cuda(index)?,
        "metal" => Device::new_metal(index)?,
        other => anyhow::bail!("unknown device '{other}' (cpu, cuda, cuda:N, metal)"),
    })
}

impl ModelArgs {
    pub fn load(&self) -> Result<TTSModel> {
        let device = parse_device(&self.device)?;
        let mut model = match &self.model {
            Some(path) => TTSModel::load_gguf(path, &device)?,
            None => {
                let mut m = TTSModel::load_with_params_device(
                    &self.variant,
                    pocket_tts::config::defaults::TEMPERATURE,
                    self.lsd_decode_steps,
                    self.eos_threshold,
                    self.noise_clamp,
                    &device,
                )?;
                m.variant = self.variant.clone();
                m
            }
        };
        model.temp = self.temperature.unwrap_or(model.config.default_temperature);
        model.lsd_decode_steps = self.lsd_decode_steps;
        model.eos_threshold = self.eos_threshold;
        model.noise_clamp = self.noise_clamp;
        model.normalize_text = !self.no_normalize;
        model.tighten_pauses = self.tight_pauses;
        Ok(model)
    }

    /// Short description for logs.
    pub fn describe(&self) -> String {
        match &self.model {
            Some(p) => p.display().to_string(),
            None => self.variant.clone(),
        }
    }
}
