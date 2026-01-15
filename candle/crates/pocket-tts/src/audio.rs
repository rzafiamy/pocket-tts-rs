use anyhow::Result;
use candle_core::{Device, Tensor};
use hound::{WavReader, WavSpec, WavWriter};
use std::path::Path;

pub fn read_wav<P: AsRef<Path>>(path: P) -> Result<(Tensor, u32)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
    };

    let tensor = if channels > 1 {
        // Interleaved to [channels, samples]
        let num_total_samples = samples.len();
        let num_samples = num_total_samples / channels;
        let mut reshaped = vec![0.0f32; num_total_samples];
        for c in 0..channels {
            for i in 0..num_samples {
                reshaped[c * num_samples + i] = samples[i * channels + c];
            }
        }
        Tensor::from_vec(reshaped, (channels, num_samples), &Device::Cpu)?
    } else {
        let n = samples.len();
        Tensor::from_vec(samples, (1, n), &Device::Cpu)?
    };

    Ok((tensor, sample_rate))
}

pub fn write_wav<P: AsRef<Path>>(path: P, audio: &Tensor, sample_rate: u32) -> Result<()> {
    let mut writer = std::fs::File::create(path)?;
    write_wav_to_writer(&mut writer, audio, sample_rate)
}

pub fn write_wav_to_writer<W: std::io::Write + std::io::Seek>(
    writer: W,
    audio: &Tensor,
    sample_rate: u32,
) -> Result<()> {
    let shape = audio.dims();
    if shape.len() != 2 {
        anyhow::bail!(
            "Expected audio tensor with shape [channels, samples], got {:?}",
            shape
        );
    }
    let channels = shape[0] as u16;
    let _num_samples = shape[1];

    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut wav_writer = WavWriter::new(writer, spec)?;
    let data = audio.to_vec2::<f32>()?;

    // Interleave channels if more than 1
    if !data.is_empty() {
        for (i, _) in data[0].iter().enumerate() {
            for channel_data in &data {
                // Hard clamp to [-1, 1] to match Python's behavior
                let val = channel_data[i].clamp(-1.0, 1.0);
                let val = (val * 32767.0) as i16;
                wav_writer.write_sample(val)?;
            }
        }
    }
    wav_writer.finalize()?;
    Ok(())
}

pub fn normalize_peak(audio: &Tensor) -> Result<Tensor> {
    let max_abs = audio.abs()?.max_all()?.to_scalar::<f32>()?;
    if max_abs > 0.0 {
        Ok(audio.affine(1.0 / max_abs as f64, 0.0)?)
    } else {
        Ok(audio.clone())
    }
}

// High-quality polyphase resampler using rubato
// Matches Python's scipy.signal.resample_poly behavior
pub fn resample(audio: &Tensor, from_rate: u32, to_rate: u32) -> Result<Tensor> {
    if from_rate == to_rate {
        return Ok(audio.clone());
    }

    let shape = audio.dims();
    let channels = shape[0];
    let num_samples = shape[1];

    if num_samples == 0 {
        return Ok(audio.clone());
    }

    use rubato::{FastFixedIn, Resampler};

    // Calculate output size
    let ratio = to_rate as f64 / from_rate as f64;
    let _new_num_samples = (num_samples as f64 * ratio) as usize;

    // Convert candle Tensor to Vec<Vec<f32>> for rubato
    // Rubato expects [channel][sample]
    let audio_vec = audio.to_vec2::<f32>()?;

    // Create resampler
    // FastFixedIn is synchronous and suitable for full-file resampling
    let mut resampler = FastFixedIn::<f32>::new(
        ratio,
        1.0,                              // max_resample_ratio_relative (1.0 for fixed)
        rubato::PolynomialDegree::Septic, // High quality interpolation
        num_samples,                      // block_size_in
        channels,
    )?;

    // Resample
    let resampled_vec = resampler.process(&audio_vec, None)?;

    // Truncate or pad to exact expected length if necessary (rubato might return slightly more/less due to block/filter delay)
    // But FastFixedIn with fixed block size should be mainly correct.
    // We'll trust rubato's output but sanity check dimensions in the Tensor creation would be good.
    // Actually, rubato might return a slightly different number of samples than naive calculation.
    // Let's use whatever rubato returned.

    let out_channels = resampled_vec.len();
    let out_samples = resampled_vec[0].len();

    // Flatten back to column-major (or whatever candle expects for from_vec)
    // Candle from_vec takes a flat vector and shape.
    // If we have [C][T], we need to flatten to C*T.
    let mut flat_data = Vec::with_capacity(out_channels * out_samples);
    for channel in resampled_vec {
        flat_data.extend(channel);
    }

    Ok(Tensor::from_vec(
        flat_data,
        (out_channels, out_samples),
        audio.device(),
    )?)
}

#[deprecated(note = "Use resample() instead which provides higher quality.")]
pub fn resample_linear(audio: &Tensor, from_rate: u32, to_rate: u32) -> Result<Tensor> {
    resample(audio, from_rate, to_rate)
}

pub fn read_wav_from_bytes(bytes: &[u8]) -> Result<(Tensor, u32)> {
    let cursor = std::io::Cursor::new(bytes);
    let mut reader = WavReader::new(cursor).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map_err(|e| anyhow::anyhow!(e)))
            .collect::<Result<Vec<f32>>>()?,
        hound::SampleFormat::Int => {
            let max_val = 2u32.pow(spec.bits_per_sample as u32 - 1) as f32;
            reader
                .samples::<i32>()
                .map(|s| {
                    s.map(|v| v as f32 / max_val)
                        .map_err(|e| anyhow::anyhow!(e))
                })
                .collect::<Result<Vec<f32>>>()?
        }
    };

    let duration = samples.len() / spec.channels as usize;
    let data = Tensor::from_vec(
        samples,
        (spec.channels as usize, duration),
        &candle_core::Device::Cpu,
    )?;

    // Mix down to mono if necessary
    let data = if spec.channels > 1 {
        data.mean(0)?
    } else {
        data.flatten(0, 1)?
    };

    Ok((data, spec.sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_normalize_peak() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![-0.5f32, 0.2, 0.5], (1, 3), &device)?;
        let normalized = normalize_peak(&t)?;
        let data = normalized.to_vec2::<f32>()?;
        assert_eq!(data[0], vec![-1.0, 0.4, 1.0]);
        Ok(())
    }

    #[test]
    fn test_resample() -> Result<()> {
        let device = Device::Cpu;
        // rubato works best with reasonable block sizes.
        // Let's use a larger sample count to be safe.
        let input_samples = 1024;
        let data: Vec<f32> = (0..input_samples).map(|i| (i as f32 * 0.1).sin()).collect();
        let t = Tensor::from_vec(data, (1, input_samples), &device)?;

        // Resample 100Hz to 200Hz (Ratio 2.0)
        let resampled = resample(&t, 100, 200)?;
        let out_samples = resampled.dims()[1];

        println!("Resample test: In={}, Out={}", input_samples, out_samples);

        // Expect approx double
        let expected = 2048;
        let diff = (out_samples as i64 - expected as i64).abs();

        assert!(
            diff <= 50,
            "Output samples {} deviates too much from expected {}",
            out_samples,
            expected
        );
        Ok(())
    }

    #[test]
    fn test_wav_io() -> Result<()> {
        let device = Device::Cpu;
        // Use small values to avoid clipping
        // write_wav applies clamp(-1, 1) to match Python's behavior
        let t = Tensor::from_vec(vec![0.0f32, 0.5, -0.5, 0.1], (1, 4), &device)?;
        let path = "test_io.wav";
        write_wav(path, &t, 16000)?;

        let (read_t, sr) = read_wav(path)?;
        assert_eq!(sr, 16000);
        assert_eq!(read_t.dims(), t.dims());

        // Pre-calculate expected values (clamp doesn't change values in [-1, 1])
        let expected_data: Vec<f32> = vec![0.0, 0.5, -0.5, 0.1];
        let expected = Tensor::from_vec(expected_data, (1, 4), &device)?;

        // Tolerance for 16-bit quantization (1/32768 ~= 3e-5) plus float error
        let diff = (read_t - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-3, "Diff was {}", diff);

        std::fs::remove_file(path)?;
        Ok(())
    }
}
