//! Splits generation time between the FlowLM backbone + sampler head and the
//! Mimi decoder, for one chunk, mirroring `generate_stream_segment`.
//! Usage: cargo run --release --example profile -- <variant> [cuda]
use candle_core::{DType, Device, Module, Tensor};
use pocket_tts::TTSModel;
use pocket_tts::voice_state::init_states;
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let variant = args.get(1).map(String::as_str).unwrap_or("english");
    let device = if args.get(2).is_some_and(|d| d == "cuda") {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let model = load(variant, &device)?;
    let variant = model.variant.as_str();
    let voice = pocket_tts::voices::default_voice(variant);
    let voice_state = match model.embedded_voices.get(voice) {
        Some(s) => s.clone(),
        None => {
            let url = pocket_tts::voices::predefined_voice_url(variant, voice);
            model.get_voice_state_from_prompt_file(pocket_tts::weights::download_if_necessary(
                &url,
            )?)?
        }
    };

    let text = "Early systems focused on symbolic reasoning, while modern methods learn from data.";
    for run in 0..3 {
        let mut state = voice_state.clone();
        let mut mimi_state = init_states(1, 1000);
        let tokens = model.conditioner.prepare(text, &device)?;
        let emb = model.conditioner.forward(&tokens)?;

        let t = Instant::now();
        model.flow_lm.transformer.forward(&emb, &mut state, 0)?;
        sync(&device)?;
        let prompt = t.elapsed();

        let time_emb = model
            .flow_lm
            .flow_net
            .compute_time_embeddings(1, &device, DType::F32)?;
        let empty = Tensor::zeros((1, 0, model.dim), DType::F32, &device)?;
        let mut input = model.flow_lm.bos_emb.reshape((1, 1, model.ldim))?;
        let (mut t_lm, mut t_mimi) = (Duration::ZERO, Duration::ZERO);
        let mut t_parts = [Duration::ZERO; 3];
        let frames = 60;
        for step in 0..frames {
            let t = Instant::now();
            let (latent, _eos) = model
                .flow_lm
                .forward(&input, &empty, &mut state, &time_emb, 0.3, -4.0, step)?;
            sync(&device)?;
            t_lm += t.elapsed();

            let t = Instant::now();
            let l = latent
                .broadcast_mul(&model.flow_lm.emb_std)?
                .broadcast_add(&model.flow_lm.emb_mean)?;
            let q = model.mimi.quantize(&l.unsqueeze(1)?.transpose(1, 2)?)?;
            let tm = Instant::now();
            let up = model
                .mimi
                .upsample
                .as_ref()
                .unwrap()
                .forward(&q, &mut mimi_state, step)?;
            sync(&device)?;
            t_parts[0] += tm.elapsed();
            let tm = Instant::now();
            let tr = model
                .mimi
                .decoder_transformer
                .forward(&up, &mut mimi_state, step)?
                .remove(0);
            sync(&device)?;
            t_parts[1] += tm.elapsed();
            let tm = Instant::now();
            let _audio = model.mimi.decoder.forward(&tr, &mut mimi_state, step)?;
            sync(&device)?;
            t_parts[2] += tm.elapsed();
            t_mimi += t.elapsed();
            input = latent.unsqueeze(1)?;
        }
        let audio_s = frames as f64 / 12.5;
        let total = prompt + t_lm + t_mimi;
        // FlowLM breakdown on a copy of the state.
        let mut st2 = state.clone();
        let x = model.flow_lm.input_linear.forward(&input)?;
        let (mut t_tr, mut t_head) = (Duration::ZERO, Duration::ZERO);
        for step in 0..20 {
            let t = Instant::now();
            let out = model.flow_lm.transformer.forward(&x, &mut st2, step)?;
            sync(&device)?;
            t_tr += t.elapsed();
            let t = Instant::now();
            let last = model.flow_lm.out_norm.forward(&out)?.squeeze(1)?;
            let c = model.flow_lm.flow_net.embed_condition(&last)?;
            let mods = model
                .flow_lm
                .flow_net
                .precompute_modulations(&c, &time_emb)?;
            let noise = Tensor::zeros((1, model.ldim), DType::F32, &device)?;
            let _ = pocket_tts::models::flow_lm::sampler_decode(
                &model.flow_lm.flow_net,
                &mods,
                &noise,
            )?;
            sync(&device)?;
            t_head += t.elapsed();
        }
        println!(
            "        transformer {:.2} ms/frame | sampler head {:.2} ms/frame",
            t_tr.as_secs_f64() * 1e3 / 20.0,
            t_head.as_secs_f64() * 1e3 / 20.0
        );
        println!(
            "        mimi: upsample {:.2} | transformer {:.2} | seanet {:.2} ms/frame",
            t_parts[0].as_secs_f64() * 1e3 / frames as f64,
            t_parts[1].as_secs_f64() * 1e3 / frames as f64,
            t_parts[2].as_secs_f64() * 1e3 / frames as f64
        );
        println!(
            "run {run}: prompt {:.1} ms | flow_lm {:.2} ms/frame | mimi {:.2} ms/frame | {:.2}x real-time",
            prompt.as_secs_f64() * 1e3,
            t_lm.as_secs_f64() * 1e3 / frames as f64,
            t_mimi.as_secs_f64() * 1e3 / frames as f64,
            audio_s / total.as_secs_f64()
        );
    }
    Ok(())
}

fn sync(device: &Device) -> anyhow::Result<()> {
    device.synchronize()?;
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
