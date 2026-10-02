//! Main TTSModel struct - orchestrates the TTS pipeline
//!
//! This is the high-level API for text-to-speech generation,
//! matching Python's `pocket_tts/models/tts_model.py`.

use crate::ModelState;
use crate::conditioners::text::LUTConditioner;
use crate::config::{Config, defaults, load_config};
use crate::models::flow_lm::FlowLMModel;
use crate::models::mimi::MimiModel;
use crate::models::seanet::{SEANetDecoder, SEANetEncoder};
use crate::models::transformer::{ProjectedTransformer, StreamingTransformer};
use crate::modules::mlp::SimpleMLPAdaLN;
use crate::voice_state::{increment_steps, init_states};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

/// Main TTS model that orchestrates the entire pipeline
#[derive(Clone)]
pub struct TTSModel {
    /// Flow language model for latent generation
    pub flow_lm: FlowLMModel,
    /// Mimi neural audio codec
    pub mimi: MimiModel,
    /// Text conditioner (tokenizer + embeddings)
    pub conditioner: LUTConditioner,
    /// Speaker projection weight for voice cloning
    pub speaker_proj_weight: Tensor,
    /// Learned embedding prepended to voice prompts (newer models)
    pub bos_before_voice: Option<Tensor>,
    /// Loaded configuration (text preparation options, defaults)
    pub config: Config,
    /// Config name the model was loaded from (`english`, `french_24l`, ...);
    /// selects the predefined voices. Empty when loaded from bytes.
    pub variant: String,
    /// Voices shipped inside a GGUF model file, by name.
    pub embedded_voices: std::sync::Arc<std::collections::HashMap<String, ModelState>>,
    /// Frames generated after EOS; `None` uses the model's recommendation or
    /// a guess from the chunk length.
    pub frames_after_eos: Option<usize>,
    /// Generation temperature
    pub temp: f32,
    /// Number of LSD decode steps
    pub lsd_decode_steps: usize,
    /// End-of-sequence threshold
    pub eos_threshold: f32,
    pub noise_clamp: Option<f32>,
    /// Spell out numbers, times, amounts, units and abbreviations and strip
    /// Markdown before synthesis (`normalize.rs`); the model was trained on
    /// spelled-out text. Number rules exist for French and English.
    pub normalize_text: bool,
    /// Optional override for voice-conditioning Mimi chunk size (in frames).
    /// If `None`, an adaptive heuristic is used.
    pub voice_prompt_chunk_frames: Option<usize>,
    /// Sample rate
    pub sample_rate: usize,
    /// Model dimension
    pub dim: usize,
    /// Latent dimension
    pub ldim: usize,
    /// Device
    pub device: Device,
}

impl TTSModel {
    /// Load a pre-trained TTS model from HuggingFace
    ///
    /// # Arguments
    /// * `variant` - Model variant (e.g., "b6369a24")
    ///
    /// # Returns
    /// Fully initialized TTSModel ready for generation
    pub fn load(variant: &str) -> Result<Self> {
        Self::load_with_params(
            variant,
            defaults::TEMPERATURE,
            defaults::LSD_DECODE_STEPS,
            defaults::EOS_THRESHOLD,
        )
    }

    /// Load with custom generation parameters
    pub fn load_with_params(
        variant: &str,
        temp: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
    ) -> Result<Self> {
        Self::load_with_params_device(
            variant,
            temp,
            lsd_decode_steps,
            eos_threshold,
            None,
            &Device::Cpu,
        )
    }

    /// Load with custom generation parameters and specific device
    pub fn load_with_params_device(
        variant: &str,
        temp: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
        noise_clamp: Option<f32>,
        device: &Device,
    ) -> Result<Self> {
        let config = resolve_config(variant)?;

        let mut model = Self::from_config(
            config,
            temp,
            lsd_decode_steps,
            eos_threshold,
            noise_clamp,
            device,
        )?;
        model.variant = variant.to_string();
        Ok(model)
    }

    /// Create model from configuration
    fn from_config(
        config: Config,
        temp: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
        noise_clamp: Option<f32>,
        device: &Device,
    ) -> Result<Self> {
        let dtype = DType::F32;

        // Download weights
        {
            let weights_path = config
                .weights_path
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("weights_path not specified in config"))?;
            // The voice-cloning repo is gated; like the Python reference, fall
            // back to the ungated weights (predefined voices only) on failure.
            let weights_file = match crate::weights::download_if_necessary(weights_path) {
                Ok(f) => f,
                Err(e) => match &config.weights_path_without_voice_cloning {
                    Some(fallback) => {
                        tracing::warn!(
                            "voice-cloning weights unavailable ({e}); using weights without voice cloning"
                        );
                        crate::weights::download_if_necessary(fallback)?
                    }
                    None => return Err(e),
                },
            };

            // Load safetensors with VarBuilder
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[weights_file], dtype, device)? };

            // Download tokenizer
            let tokenizer_path =
                crate::weights::download_if_necessary(&config.flow_lm.lookup_table.tokenizer_path)?;

            // Build conditioner
            let conditioner = LUTConditioner::new(
                config.flow_lm.lookup_table.n_bins,
                &tokenizer_path,
                config.flow_lm.lookup_table.dim,
                config.flow_lm.transformer.d_model,
                vb.pp("flow_lm.conditioner"),
            )?;

            Self::from_config_and_vb(
                config,
                temp,
                lsd_decode_steps,
                eos_threshold,
                noise_clamp,
                conditioner,
                vb,
            )
        }
    }

    /// Load a single-file GGUF model written by `gguf::convert`.
    pub fn load_gguf<P: AsRef<std::path::Path>>(path: P, device: &Device) -> Result<Self> {
        let g = crate::gguf::read(path.as_ref(), device)?;
        let vb = VarBuilder::from_tensors(g.dense, DType::F32, device);
        let _scope = crate::modules::linear::QuantScope::enter(g.quant);
        let conditioner = LUTConditioner::new_from_bytes(
            g.config.flow_lm.lookup_table.n_bins,
            g.tokenizer_json.as_bytes(),
            g.config.flow_lm.lookup_table.dim,
            g.config.flow_lm.transformer.d_model,
            vb.pp("flow_lm.conditioner"),
        )?;
        let temp = g.config.default_temperature;
        let mut model = Self::from_config_and_vb(
            g.config,
            temp,
            defaults::LSD_DECODE_STEPS,
            defaults::EOS_THRESHOLD,
            None,
            conditioner,
            vb,
        )?;
        model.variant = g.variant;
        model.embedded_voices = std::sync::Arc::new(g.voices);
        Ok(model)
    }

    /// Load model from byte slices (useful for WASM)
    pub fn load_from_bytes(
        config_yaml: &[u8],
        weights_bytes: &[u8],
        tokenizer_bytes: &[u8],
    ) -> Result<Self> {
        let config: Config = serde_yaml::from_slice(config_yaml)?;
        let device = Device::Cpu;
        let dtype = DType::F32;

        let tensors = candle_core::safetensors::load_buffer(weights_bytes, &device)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);

        // On WASM, LUTConditioner::new needs a path, but we've updated it to
        // eventually support bytes. For now, we'll need to adapt it.
        // Actually, my recent change to conditioners/text.rs still uses Tokenizer::from_file on WASM.
        // I should probably fix that to support bytes too.

        // For now, let's keep it simple and assume we have a path for the tokenizer or a way to load it.
        // This is a placeholder for real WASM loading.

        let conditioner = LUTConditioner::new_from_bytes(
            config.flow_lm.lookup_table.n_bins,
            tokenizer_bytes,
            config.flow_lm.lookup_table.dim,
            config.flow_lm.transformer.d_model,
            vb.pp("flow_lm.conditioner"),
        )?;

        Self::from_config_and_vb(
            config,
            defaults::TEMPERATURE,
            defaults::LSD_DECODE_STEPS,
            defaults::EOS_THRESHOLD,
            None,
            conditioner,
            vb,
        )
    }

    /// Internal helper to build model from config and VarBuilder
    fn from_config_and_vb(
        config: Config,
        temp: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
        noise_clamp: Option<f32>,
        conditioner: LUTConditioner,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();

        // Build FlowLM components
        let dim = config.flow_lm.transformer.d_model;
        let ldim = config.mimi.quantizer.dimension;
        let hidden_dim = dim * config.flow_lm.transformer.hidden_scale;

        let num_time_conds = match config.flow_lm.flow.flow_type.as_str() {
            "lsd" => 2,
            "flow_matching" => 1,
            "drifting" => 0,
            other => anyhow::bail!("Unknown flow type: {other}"),
        };
        let flow_net = SimpleMLPAdaLN::new(
            ldim,                      // in_channels (input is latent dim)
            config.flow_lm.flow.dim,   // model_channels
            ldim,                      // out_channels (output is also latent dim)
            dim,                       // cond_channels (conditioning from transformer)
            config.flow_lm.flow.depth, // num_res_blocks
            num_time_conds,
            config.flow_lm.transformer.max_period as f32,
            vb.pp("flow_lm.flow_net"),
        )?;

        // StreamingTransformer::new(d_model, num_heads, num_layers, layer_scale, dim_feedforward, context, max_period, kind, name, vb)
        let transformer = StreamingTransformer::new(
            dim,
            config.flow_lm.transformer.num_heads,
            config.flow_lm.transformer.num_layers,
            None,       // layer_scale
            hidden_dim, // dim_feedforward
            None,       // context (causal)
            config.flow_lm.transformer.max_period as f32,
            "kv",
            "flow_lm.transformer",
            vb.pp("flow_lm.transformer"),
        )?;

        let mut flow_lm = FlowLMModel::new(flow_net, transformer, ldim, dim, vb.pp("flow_lm"))?;
        flow_lm.noise_clamp = noise_clamp;

        // Build Mimi components
        let seanet_cfg = &config.mimi.seanet;
        let encoder = SEANetEncoder::new(
            seanet_cfg.channels,
            seanet_cfg.dimension,
            seanet_cfg.n_filters,
            seanet_cfg.n_residual_layers,
            &seanet_cfg.ratios,
            seanet_cfg.kernel_size,
            seanet_cfg.residual_kernel_size,
            seanet_cfg.last_kernel_size,
            seanet_cfg.dilation_base,
            &seanet_cfg.pad_mode,
            seanet_cfg.compress,
            "mimi.encoder",
            vb.pp("mimi.encoder"),
        )?;

        let decoder = SEANetDecoder::new(
            seanet_cfg.channels,
            seanet_cfg.dimension,
            seanet_cfg.n_filters,
            seanet_cfg.n_residual_layers,
            &seanet_cfg.ratios,
            seanet_cfg.kernel_size,
            seanet_cfg.residual_kernel_size,
            seanet_cfg.last_kernel_size,
            seanet_cfg.dilation_base,
            &seanet_cfg.pad_mode,
            seanet_cfg.compress,
            "mimi.decoder",
            vb.pp("mimi.decoder"),
        )?;

        let mimi_tr_cfg = &config.mimi.transformer;
        // ProjectedTransformer::new(input_dimension, output_dimensions, d_model, num_heads, num_layers, layer_scale, context, max_period, dim_feedforward, name, vb)
        let encoder_transformer = ProjectedTransformer::new(
            mimi_tr_cfg.input_dimension,
            mimi_tr_cfg.output_dimensions.clone(),
            mimi_tr_cfg.d_model,
            mimi_tr_cfg.num_heads,
            mimi_tr_cfg.num_layers,
            mimi_tr_cfg.layer_scale as f32,
            mimi_tr_cfg.context,
            mimi_tr_cfg.max_period as f32,
            mimi_tr_cfg.dim_feedforward,
            "mimi.encoder_transformer",
            vb.pp("mimi.encoder_transformer"),
        )?;

        let decoder_transformer = ProjectedTransformer::new(
            mimi_tr_cfg.input_dimension,
            mimi_tr_cfg.output_dimensions.clone(),
            mimi_tr_cfg.d_model,
            mimi_tr_cfg.num_heads,
            mimi_tr_cfg.num_layers,
            mimi_tr_cfg.layer_scale as f32,
            mimi_tr_cfg.context,
            mimi_tr_cfg.max_period as f32,
            mimi_tr_cfg.dim_feedforward,
            "mimi.decoder_transformer",
            vb.pp("mimi.decoder_transformer"),
        )?;

        // Calculate encoder frame rate from SEANet ratios
        let hop_length: usize = seanet_cfg.ratios.iter().product();
        let encoder_frame_rate = config.mimi.sample_rate as f64 / hop_length as f64;

        let mimi = MimiModel::new(
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            config.mimi.frame_rate,
            encoder_frame_rate,
            config.mimi.sample_rate,
            config.mimi.channels,
            config.mimi.quantizer.dimension,
            config.mimi.quantizer.output_dimension,
            seanet_cfg.dimension,
            config.mimi.inner_dim,
            config.mimi.outer_dim,
            "mimi",
            vb.pp("mimi"),
        )?;

        // Speaker projection maps encoder latents to the backbone: [dim, inner_dim]
        // (32 for newer models, the 512-channel seanet dimension for b6369a24).
        let latent_dim = config.mimi.inner_dim.unwrap_or(seanet_cfg.dimension);
        let speaker_proj_weight = vb.get((dim, latent_dim), "flow_lm.speaker_proj_weight")?;
        let bos_before_voice = if config.flow_lm.insert_bos_before_voice {
            Some(vb.get((1, 1, dim), "flow_lm.bos_before_voice")?)
        } else {
            None
        };

        Ok(Self {
            flow_lm,
            mimi,
            conditioner,
            speaker_proj_weight,
            bos_before_voice,
            sample_rate: config.mimi.sample_rate,
            config,
            variant: String::new(),
            embedded_voices: Default::default(),
            frames_after_eos: None,
            temp,
            lsd_decode_steps,
            eos_threshold,
            noise_clamp,
            voice_prompt_chunk_frames: None,
            normalize_text: true,
            dim,
            ldim,
            device,
        })
    }

    /// Create voice state from audio prompt bytes for voice cloning
    pub fn get_voice_state_from_bytes(&self, bytes: &[u8]) -> Result<ModelState> {
        let (audio, sample_rate) = crate::audio::read_wav_from_bytes(bytes)?;

        // Resample to model sample rate if needed
        let audio = if sample_rate != self.sample_rate as u32 {
            crate::audio::resample(&audio, sample_rate, self.sample_rate as u32)?
        } else {
            audio
        };

        // Add batch dimension: [C, T] -> [B, C, T]
        let audio = audio.unsqueeze(0)?;

        self.get_voice_state_from_tensor(&audio)
    }

    /// Create voice state from audio prompt for voice cloning
    ///
    /// Encodes the audio through Mimi and projects to flow model space.
    pub fn get_voice_state<P: AsRef<std::path::Path>>(&self, audio_path: P) -> Result<ModelState> {
        let (audio, sample_rate) = crate::audio::read_wav(audio_path)?;

        // Resample to model sample rate if needed
        let audio = if sample_rate != self.sample_rate as u32 {
            crate::audio::resample(&audio, sample_rate, self.sample_rate as u32)?
        } else {
            audio
        };

        // Add batch dimension: [C, T] -> [B, C, T]
        let audio = audio.unsqueeze(0)?;

        self.get_voice_state_from_tensor(&audio)
    }

    /// Create voice state from a .safetensors voice: either a latent prompt
    /// (`audio_prompt`) or an exported model state (`<module>/<key>`, the
    /// format of `pocket-tts export-voice` and the per-language voices).
    pub fn get_voice_state_from_prompt_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<ModelState> {
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        self.voice_state_from_tensors(tensors)
    }

    /// Create voice state from .safetensors bytes (see `get_voice_state_from_prompt_file`)
    pub fn get_voice_state_from_prompt_bytes(&self, bytes: &[u8]) -> Result<ModelState> {
        let tensors = candle_core::safetensors::load_buffer(bytes, &self.device)?;
        self.voice_state_from_tensors(tensors)
    }

    fn voice_state_from_tensors(
        &self,
        tensors: std::collections::HashMap<String, Tensor>,
    ) -> Result<ModelState> {
        if let Some(prompt) = tensors.get("audio_prompt") {
            return self.get_voice_state_from_prompt_tensor(prompt);
        }
        if tensors.keys().any(|k| k.contains('/')) {
            return import_model_state(&tensors, &self.device);
        }
        anyhow::bail!("voice file holds neither 'audio_prompt' nor an exported model state")
    }

    /// Create voice state from a pre-calculated latent prompt tensor
    pub fn get_voice_state_from_prompt_tensor(&self, prompt: &Tensor) -> Result<ModelState> {
        // Ensure prompt tensor is on the same device as the model (fixes Metal device mismatch)
        let prompt = if prompt.device().same_device(&self.device) {
            prompt.clone()
        } else {
            prompt.to_device(&self.device)?
        };

        let mut flow_state = init_states(1, 1000);
        self.run_flow_lm_prompt(&prompt, &mut flow_state)?;
        Ok(flow_state)
    }

    /// Create voice state from audio tensor
    pub fn get_voice_state_from_tensor(&self, audio: &Tensor) -> Result<ModelState> {
        let mut model_state = init_states(1, 1000);

        // Ensure audio tensor is on the same device as the model (fixes Metal device mismatch)
        let audio = if audio.device().same_device(&self.device) {
            audio.clone()
        } else {
            audio.to_device(&self.device)?
        };

        // Training prompts end inside a pause between words.
        let audio =
            crate::audio::end_on_pause(&audio.squeeze(0)?, self.sample_rate)?.unsqueeze(0)?;

        // Pad audio to a multiple of frame size for streaming conv stride alignment
        let frame_size = self.mimi.frame_size();
        let (b, c, t) = audio.dims3()?;
        let pad_len = if t % frame_size != 0 {
            frame_size - (t % frame_size)
        } else {
            0
        };
        let audio = if pad_len > 0 {
            // Create padding on the same device as audio (which is now on self.device)
            let pad = Tensor::zeros((b, c, pad_len), audio.dtype(), &self.device)?;
            Tensor::cat(&[&audio, &pad], 2)?
        } else {
            audio
        };

        // Encode audio through Mimi in chunks to avoid OOM in SEANet Conv1d layers.
        // Chunk size is adapted to prompt length unless explicitly overridden.
        let mut encoded_chunks = Vec::new();
        let (_b, _c, total_samples) = audio.dims3()?;
        let chunk_frames = self.adaptive_voice_prompt_chunk_frames(total_samples, frame_size);
        let chunk_size = frame_size * chunk_frames;

        for start in (0..total_samples).step_by(chunk_size) {
            let end = std::cmp::min(start + chunk_size, total_samples);
            let chunk = audio.narrow(2, start, end - start)?;
            let code = self.mimi.encode_to_latent(&chunk, &mut model_state, 0)?;
            encoded_chunks.push(code);
        }
        let encoded = Tensor::cat(&encoded_chunks, 2)?;

        // Transpose from [B, D, T] to [B, T, D]
        let latents = encoded.transpose(1, 2)?.to_dtype(DType::F32)?;

        // Project to flow model space: [B, T, ldim] @ [dim, ldim].T -> [B, T, dim]
        // Candle needs 2D @ 2D for matmul, so reshape
        let (b, t, d) = latents.dims3()?;
        let latents_2d = latents.reshape((b * t, d))?;
        let conditioning_2d = latents_2d.matmul(&self.speaker_proj_weight.t()?)?;
        let conditioning = conditioning_2d.reshape((b, t, self.dim))?;
        let conditioning = match &self.bos_before_voice {
            Some(bos) => Tensor::cat(&[bos, &conditioning], 1)?,
            None => conditioning,
        };

        // Run flow_lm with audio conditioning to update state
        let mut flow_state = init_states(1, 1000);
        self.run_flow_lm_prompt(&conditioning, &mut flow_state)?;

        Ok(flow_state)
    }

    fn adaptive_voice_prompt_chunk_frames(&self, total_samples: usize, frame_size: usize) -> usize {
        if let Some(override_frames) = self.voice_prompt_chunk_frames {
            return override_frames.max(1);
        }

        let total_frames = total_samples.div_ceil(frame_size);
        if total_frames <= 120 {
            total_frames.max(1)
        } else if total_frames <= 600 {
            120
        } else if total_frames <= 1800 {
            180
        } else {
            240
        }
    }

    /// Run flow LM with audio conditioning (used during prompting)
    fn run_flow_lm_prompt(&self, conditioning: &Tensor, state: &mut ModelState) -> Result<()> {
        // Empty text tokens and backbone input
        let empty_text = Tensor::zeros((1, 0), DType::I64, &self.device)?;
        let text_embeddings = self.conditioner.forward(&empty_text)?;

        // Concatenate text embeddings and audio conditioning
        // Match Python/reference order: audio conditioning comes before text embeddings.
        let input = Tensor::cat(&[conditioning, &text_embeddings], 1)?;

        // Run through transformer (no generation, just prompting)
        // With custom SDPA, this is now memory efficient
        let _ = self.flow_lm.transformer.forward(&input, state, 0)?;

        // Increment FlowLM state after prompting (critical for RoPE positioning)
        // Python: increment_steps(self.flow_lm, model_state, increment=audio_conditioning.shape[1])
        let increment_by = conditioning.dims()[1];
        increment_steps(state, "offset", increment_by);

        Ok(())
    }

    /// Text preparation options from the model config.
    pub fn text_options(&self) -> crate::text_chunking::TextOptions {
        (&self.config).into()
    }

    /// Splits text into the chunks generated one at a time (Python's
    /// `split_into_best_sentences`), each at most `MAX_TOKENS_PER_CHUNK` tokens.
    pub fn split_into_best_sentences(&self, text: &str) -> Result<Vec<String>> {
        let text = crate::pause::strip_pause_markers(text);
        let text = if self.normalize_text {
            crate::normalize::normalize(&text, crate::normalize::Lang::of_variant(&self.variant))
        } else {
            text
        };
        crate::text_chunking::split_into_best_sentences(
            &self.conditioner,
            &text,
            defaults::MAX_TOKENS_PER_CHUNK,
            &self.text_options(),
        )
    }

    /// Generate audio from text with voice state
    pub fn generate(&self, text: &str, voice_state: &ModelState) -> Result<Tensor> {
        let mut audio_chunks = Vec::new();

        for chunk in self.generate_stream(text, voice_state) {
            audio_chunks.push(chunk?);
        }

        // Concatenate all audio chunks
        if audio_chunks.is_empty() {
            anyhow::bail!("No audio generated");
        }
        let audio = Tensor::cat(&audio_chunks, 2)?;
        // Remove batch dimension
        let audio = audio.squeeze(0)?;

        Ok(audio)
    }

    /// Generate audio from text with pause handling
    ///
    /// This method parses pause markers in the text and inserts silence
    /// at appropriate positions. Supports:
    /// - Explicit pauses: `[pause:500ms]` or `[pause:1s]`
    /// - Natural pauses from punctuation are handled during generation
    ///
    /// # Example
    /// ```ignore
    /// let audio = model.generate_with_pauses("Hello... [pause:500ms] world", &voice_state)?;
    /// ```
    pub fn generate_with_pauses(&self, text: &str, voice_state: &ModelState) -> Result<Tensor> {
        let mut audio_chunks = Vec::new();

        for chunk in self.generate_stream_long(text, voice_state) {
            audio_chunks.push(chunk?);
        }

        // Concatenate all audio chunks
        if audio_chunks.is_empty() {
            anyhow::bail!("No audio generated");
        }
        let audio = Tensor::cat(&audio_chunks, 2)?;
        // Remove batch dimension
        let audio = audio.squeeze(0)?;

        Ok(audio)
    }

    /// Generate audio stream from text with voice state
    ///
    /// Returns an iterator that yields audio chunks (one per Mimi frame). The
    /// text is split into sentence chunks, each generated from a copy of the
    /// voice state, as in Python's `generate_audio_stream`.
    pub fn generate_stream<'a>(
        &'a self,
        text: &str,
        voice_state: &ModelState,
    ) -> Box<dyn Iterator<Item = Result<Tensor>> + 'a> {
        let chunks = match self.split_into_best_sentences(text) {
            Ok(c) => c,
            Err(e) => return Box::new(std::iter::once(Err(e))),
        };
        let voice_state = voice_state.clone();
        Box::new(
            chunks
                .into_iter()
                .flat_map(move |chunk| self.generate_stream_segment(&chunk, &voice_state)),
        )
    }

    /// Generate audio stream from text with voice state, returning an owned iterator.
    ///
    /// This is useful for WASM bindings where the iterator must be 'static.
    pub fn generate_stream_owned(
        &self,
        text: &str,
        voice_state: &ModelState,
    ) -> Box<dyn Iterator<Item = Result<Tensor>> + 'static> {
        let model = self.clone();
        let chunks = match model.split_into_best_sentences(text) {
            Ok(c) => c,
            Err(e) => return Box::new(std::iter::once(Err(e))),
        };
        let voice_state = voice_state.clone();
        Box::new(
            chunks
                .into_iter()
                .flat_map(move |chunk| model.generate_stream_segment(&chunk, &voice_state)),
        )
    }

    /// Frames generated for `token_count` text tokens before giving up on EOS.
    fn max_gen_len(&self, token_count: usize) -> usize {
        let seconds = token_count as f64 / defaults::TOKENS_PER_SECOND_ESTIMATE
            + defaults::GEN_SECONDS_PADDING;
        (seconds * self.mimi.frame_rate).ceil() as usize
    }

    /// Generates one chunk (Python's `_generate_audio_stream_short_text`).
    fn generate_stream_segment(
        &self,
        chunk: &str,
        voice_state: &ModelState,
    ) -> Box<dyn Iterator<Item = Result<Tensor>>> {
        let fail = |e: anyhow::Error| -> Box<dyn Iterator<Item = Result<Tensor>>> {
            Box::new(std::iter::once(Err(e)))
        };
        let (prepared_text, frames_after_eos_guess) =
            match crate::text_chunking::prepare_text_prompt(chunk, &self.text_options()) {
                Ok(p) => p,
                Err(e) => return fail(e),
            };
        let frames_after_eos = self
            .frames_after_eos
            .or(self.config.model_recommended_frames_after_eos)
            .unwrap_or(frames_after_eos_guess + 2);

        let mut state = voice_state.clone();
        let mut mimi_state = init_states(1, 1000);

        let tokens = match self.conditioner.prepare(&prepared_text, &self.device) {
            Ok(t) => t,
            Err(e) => return fail(e),
        };
        let max_gen_len = self.max_gen_len(tokens.dims()[1]);
        let text_embeddings = match self.conditioner.forward(&tokens) {
            Ok(e) => e,
            Err(e) => return fail(e),
        };
        // Prompt the text; the backbone output is discarded.
        if let Err(e) = self
            .flow_lm
            .transformer
            .forward(&text_embeddings, &mut state, 0)
        {
            return fail(e.into());
        }

        let mut backbone_input = match self.flow_lm.bos_emb.reshape((1, 1, self.ldim)) {
            Ok(t) => t,
            Err(e) => return fail(e.into()),
        };
        let time_embeddings = match self.flow_lm.flow_net.compute_time_embeddings(
            self.lsd_decode_steps,
            &self.device,
            DType::F32,
        ) {
            Ok(te) => te,
            Err(e) => return fail(e.into()),
        };
        let empty_text_embeddings = match Tensor::zeros((1, 0, self.dim), DType::F32, &self.device)
        {
            Ok(t) => t,
            Err(e) => return fail(e.into()),
        };

        // A fresh Mimi decoder state starts with a small step heard as a click:
        // fade the first 5 ms of the chunk in.
        let fade_len = self.sample_rate / 200;
        let mut fade_pending = true;
        let mut eos_step: Option<usize> = None;
        let mut step = 0;
        let mut model = self.clone();
        // `noise_clamp` on the model is the setting; the sampler keeps a copy.
        model.flow_lm.noise_clamp = self.noise_clamp;

        Box::new(std::iter::from_fn(move || {
            if step >= max_gen_len {
                if step == max_gen_len && eos_step.is_none() {
                    tracing::warn!(
                        "Maximum generation length reached without EOS, this very often indicates an error."
                    );
                }
                step += 1;
                return None;
            }
            let current = step;
            step += 1;

            let (next_latent, is_eos) = match tracing::info_span!("flow_lm.forward", step = current)
                .in_scope(|| {
                    model.flow_lm.forward(
                        &backbone_input,
                        &empty_text_embeddings,
                        &mut state,
                        &time_embeddings,
                        model.temp,
                        model.eos_threshold,
                        current,
                    )
                }) {
                Ok(res) => res,
                Err(e) => {
                    step = usize::MAX;
                    return Some(Err(e.into()));
                }
            };

            if is_eos && eos_step.is_none() && current >= defaults::MIN_FRAMES_BEFORE_EOS {
                eos_step = Some(current);
            }
            if let Some(e) = eos_step
                && current >= e + frames_after_eos
            {
                step = usize::MAX;
                return None;
            }

            let frame = (|| -> Result<Tensor> {
                let latent = next_latent
                    .broadcast_mul(&model.flow_lm.emb_std)?
                    .broadcast_add(&model.flow_lm.emb_mean)?;
                let quantized = model
                    .mimi
                    .quantize(&latent.unsqueeze(1)?.transpose(1, 2)?)?;
                let audio = tracing::info_span!("mimi.decode_from_latent", step = current)
                    .in_scope(|| {
                        model
                            .mimi
                            .decode_from_latent(&quantized, &mut mimi_state, current)
                    })?;
                if fade_pending {
                    fade_pending = false;
                    return fade_in(&audio, fade_len);
                }
                Ok(audio)
            })();
            match next_latent.unsqueeze(1) {
                Ok(t) => backbone_input = t,
                Err(e) => return Some(Err(e.into())),
            }
            Some(frame)
        }))
    }

    /// Generate audio stream from text with explicit `[pause:500ms]` /
    /// `[pause:1s]` markers, which become silence between generated segments.
    pub fn generate_stream_long<'a>(
        &'a self,
        text: &str,
        voice_state: &'a ModelState,
    ) -> impl Iterator<Item = Result<Tensor>> + 'a {
        use crate::pause::{TextSegment, silence_samples, split_explicit_pauses};

        split_explicit_pauses(text)
            .into_iter()
            .flat_map(move |seg| match seg {
                TextSegment::Text(s) => self.generate_stream(&s, voice_state),
                TextSegment::Pause(ms) => {
                    let n_samples = silence_samples(ms, self.sample_rate as u32);
                    let silence =
                        Tensor::zeros((1, self.mimi.channels, n_samples), DType::F32, &self.device);
                    Box::new(std::iter::once(silence.map_err(anyhow::Error::from)))
                        as Box<dyn Iterator<Item = Result<Tensor>>>
                }
            })
    }

    /// Upper bound on generated frames for `text`, for progress reporting.
    pub fn estimate_generation_steps(&self, text: &str) -> usize {
        self.split_into_best_sentences(text)
            .unwrap_or_default()
            .iter()
            .map(|c| self.max_gen_len(self.conditioner.count_tokens(c).unwrap_or(0)))
            .sum()
    }
}

/// Converts a Python model state (`transformer.layers.N.self_attn/{cache,offset}`,
/// cache `[2, B, T, H, D]`) into this crate's FlowLM attention state.
fn import_model_state(
    tensors: &std::collections::HashMap<String, Tensor>,
    device: &Device,
) -> Result<ModelState> {
    let mut state = init_states(1, 0);
    for (key, cache) in tensors {
        let Some(module) = key.strip_suffix("/cache") else {
            continue;
        };
        let offset = if let Some(o) = tensors.get(&format!("{module}/offset")) {
            o.flatten_all()?.to_dtype(DType::I64)?.to_vec1::<i64>()?[0] as usize
        } else if let Some(end) = tensors.get(&format!("{module}/current_end")) {
            // Older exports stored the step index as current_end.shape[0].
            end.dim(0)?
        } else {
            anyhow::bail!("{module}: no offset in exported state");
        };
        if let Some(pad) = tensors.get(&format!("{module}/pad"))
            && pad.flatten_all()?.to_dtype(DType::I64)?.to_vec1::<i64>()?[0] != 0
        {
            anyhow::bail!("{module}: left-padded states are not supported");
        }
        // [2, B, T, H, D] -> two [B, H, offset, D]
        let cache = cache.to_dtype(DType::F32)?.narrow(2, 0, offset)?;
        let k = cache.get(0)?.transpose(1, 2)?.contiguous()?;
        let v = cache.get(1)?.transpose(1, 2)?.contiguous()?;
        state.insert(
            format!("flow_lm.{module}"),
            crate::voice_state::attention_state(k, v, offset, device)?,
        );
    }
    if state.is_empty() {
        anyhow::bail!("exported state has no attention caches");
    }
    Ok(state)
}

/// Config for `variant`: a built-in name (`english`, `french_24l`, ...) or a
/// path to a YAML file.
pub fn resolve_config(variant: &str) -> Result<Config> {
    if let Some(yaml) = crate::builtin_configs::get(variant) {
        return Ok(serde_yaml::from_str(yaml)?);
    }
    let path = std::path::Path::new(variant);
    if path.is_file() {
        return load_config(path);
    }
    anyhow::bail!(
        "Unknown variant '{variant}'. Built-in variants: {}",
        crate::builtin_configs::names()
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Multiplies the first `n` samples of `audio` `[B, C, T]` by a 0 -> 1 ramp
/// (`torch.linspace(0, 1, n)`).
fn fade_in(audio: &Tensor, n: usize) -> Result<Tensor> {
    let t = audio.dim(2)?;
    let n = n.min(t);
    if n == 0 {
        return Ok(audio.clone());
    }
    let ramp: Vec<f32> = (0..t)
        .map(|i| {
            if i >= n {
                1.0
            } else if n == 1 {
                0.0
            } else {
                i as f32 / (n - 1) as f32
            }
        })
        .collect();
    let ramp = Tensor::from_vec(ramp, (1, 1, t), audio.device())?.to_dtype(audio.dtype())?;
    Ok(audio.broadcast_mul(&ramp)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// covers: REQ-VOI-001
    /// Exported Python states (`<module>/cache` `[2, B, T, H, D]` + `offset`)
    /// become FlowLM attention states holding the first `offset` positions.
    #[test]
    fn import_exported_model_state() -> Result<()> {
        use crate::voice_state::{ATTN_K_BUF_KEY, ATTN_V_BUF_KEY, read_attention_cursor};
        let dev = Device::Cpu;
        let (t, h, d, offset) = (10, 2, 4, 7);
        let cache = Tensor::arange(0f32, (2 * t * h * d) as f32, &dev)?.reshape((2, 1, t, h, d))?;
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "transformer.layers.0.self_attn/cache".to_string(),
            cache.clone(),
        );
        tensors.insert(
            "transformer.layers.0.self_attn/offset".to_string(),
            Tensor::new(&[offset as i64], &dev)?,
        );
        let state = import_model_state(&tensors, &dev)?;
        let m = &state["flow_lm.transformer.layers.0.self_attn"];
        assert_eq!(m[ATTN_K_BUF_KEY].dims(), &[1, h, offset, d]);
        let expected_v = cache.get(1)?.narrow(1, 0, offset)?.transpose(1, 2)?;
        let diff = (&m[ATTN_V_BUF_KEY] - expected_v)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(diff, 0.0);
        let cursor = read_attention_cursor(m);
        assert_eq!((cursor.pos, cursor.len), (offset, offset));

        tensors.insert(
            "transformer.layers.0.self_attn/pad".to_string(),
            Tensor::new(&[1i64], &dev)?,
        );
        assert!(
            import_model_state(&tensors, &dev).is_err(),
            "padded states are refused"
        );
        Ok(())
    }

    /// covers: REQ-CFG-001
    #[test]
    fn builtin_configs_parse() {
        for name in crate::builtin_configs::names() {
            let c = resolve_config(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(c.mimi.sample_rate, 24000, "{name}");
        }
        assert_eq!(
            resolve_config("french")
                .unwrap()
                .flow_lm
                .transformer
                .num_layers,
            6
        );
        assert!(resolve_config("klingon").is_err());
    }
}
