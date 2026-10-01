//! VRAM of a bare candle CUDA context (one tiny matmul), for comparison.
fn main() -> anyhow::Result<()> {
    let d = candle_core::Device::new_cuda(0)?;
    let a = candle_core::Tensor::ones((4, 4), candle_core::DType::F32, &d)?;
    let _ = a.matmul(&a)?.to_vec2::<f32>()?;
    std::thread::sleep(std::time::Duration::from_secs(3));
    Ok(())
}
