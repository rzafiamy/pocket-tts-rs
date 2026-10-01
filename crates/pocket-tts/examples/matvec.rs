//! Batch-1 linear layer timings: candle Linear (f32) vs QMatMul (f32, f16, q8_0, q4k).
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};
use std::time::Instant;

fn bench(name: &str, f: impl Fn() -> candle_core::Result<Tensor>) -> candle_core::Result<()> {
    for _ in 0..5 {
        f()?;
    }
    let n = 200;
    let t = Instant::now();
    for _ in 0..n {
        f()?;
    }
    println!(
        "{name:>12}: {:.3} ms",
        t.elapsed().as_secs_f64() * 1e3 / n as f64
    );
    Ok(())
}

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let (din, dout) = (1024, 4096);
    let w = Tensor::randn(0f32, 0.02, (dout, din), &dev)?;
    let x = Tensor::randn(0f32, 1.0, (1, 1, din), &dev)?;
    let lin = candle_nn::Linear::new(w.clone(), None);
    bench("linear f32", || lin.forward(&x))?;
    let wt = w.t()?.contiguous()?;
    bench("matmul wT", || x.squeeze(0)?.matmul(&wt))?;
    for dt in [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::Q8_0,
        GgmlDType::Q4K,
    ] {
        let q = QMatMul::from_qtensor(QTensor::quantize(&w, dt)?)?;
        bench(&format!("qmatmul {dt:?}"), || q.forward(&x))?;
    }
    Ok(())
}
