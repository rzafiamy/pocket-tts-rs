//! Activations tuned for CPU decoding.

use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

/// sqrt(2/pi)
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// GELU, tanh approximation (`F.gelu(x, approximate="tanh")`).
///
/// Uses `0.5 * (1 + tanh(z)) = sigmoid(2z)`: one `exp` per element instead
/// of `tanh`, which libm evaluates about eight times slower.
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 {
        return x.apply_op1_no_bwd(&GeluTanh);
    }
    x.gelu()
}

struct GeluTanh;

#[inline]
fn gelu_one(x: f32) -> f32 {
    let u = 2.0 * SQRT_2_OVER_PI * x * (1.0 + 0.044_715 * x * x);
    x / (1.0 + (-u).exp())
}

impl CustomOp1 for GeluTanh {
    fn name(&self) -> &'static str {
        "gelu-tanh-exp"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(data) = storage else {
            candle_core::bail!("gelu_tanh: f32 only");
        };
        let src = match layout.contiguous_offsets() {
            Some((start, end)) => &data[start..end],
            None => candle_core::bail!("gelu_tanh: input must be contiguous"),
        };
        const CHUNK: usize = 16 * 1024;
        let mut out = vec![0f32; src.len()];
        if src.len() > CHUNK {
            out.par_chunks_mut(CHUNK)
                .zip(src.par_chunks(CHUNK))
                .for_each(|(o, s)| o.iter_mut().zip(s).for_each(|(o, &x)| *o = gelu_one(x)));
        } else {
            out.iter_mut().zip(src).for_each(|(o, &x)| *o = gelu_one(x));
        }
        Ok((CpuStorage::F32(out), layout.shape().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// covers: REQ-OPS-001
    #[test]
    fn matches_candle_gelu() -> Result<()> {
        let x = Tensor::randn(0f32, 3.0, (4, 40_000), &Device::Cpu)?;
        let diff = (gelu_tanh(&x)? - x.gelu()?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5, "max diff {diff}");
        let big = Tensor::new(&[-100f32, -10.0, 0.0, 10.0, 100.0], &Device::Cpu)?;
        let v = gelu_tanh(&big)?.to_vec1::<f32>()?;
        assert_eq!(v[0], -0.0);
        assert_eq!(v[2], 0.0);
        assert!((v[4] - 100.0).abs() < 1e-4);
        Ok(())
    }
}
