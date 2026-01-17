use candle_core::{DType, Device, Tensor};
use candle_nn::ops::sdpa;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let q = Tensor::randn(0f32, 1f32, (1, 1, 10, 64), &device)?;
    let k = Tensor::randn(0f32, 1f32, (1, 1, 10, 64), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, 1, 10, 64), &device)?;

    // Check if sdpa is available on CPU
    let out = sdpa(&q, &k, &v, 1.0 / 8.0, 0.0)?;
    println!("SDPA output shape: {:?}", out.shape());
    Ok(())
}
