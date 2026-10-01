//! Conv timings at Mimi decoder shapes (one 80 ms frame).
use candle_core::{Device, Tensor};
use std::time::Instant;

fn bench(name: &str, f: impl Fn() -> candle_core::Result<Tensor>) -> candle_core::Result<()> {
    for _ in 0..3 {
        f()?;
    }
    let n = 50;
    let t = Instant::now();
    for _ in 0..n {
        f()?;
    }
    println!(
        "{name:>40}: {:.3} ms",
        t.elapsed().as_secs_f64() * 1e3 / n as f64
    );
    Ok(())
}

/// conv1d (stride 1, no padding) as im2col + one matmul.
fn conv_mm(x: &Tensor, w2: &Tensor, k: usize, dilation: usize) -> candle_core::Result<Tensor> {
    let (_b, c, t) = x.dims3()?;
    let t_out = t - dilation * (k - 1);
    // cols[c*k + j, t] = x[c, t + j*dilation]  -> (C*k, T_out)
    let x0 = x.squeeze(0)?;
    let cols: Vec<Tensor> = (0..k)
        .map(|j| x0.narrow(1, j * dilation, t_out))
        .collect::<candle_core::Result<_>>()?;
    let cols = Tensor::stack(&cols, 1)?.reshape((c * k, t_out))?;
    w2.matmul(&cols)?.unsqueeze(0)
}

fn main() -> candle_core::Result<()> {
    let d = Device::Cpu;
    // upsample: depthwise convtr 512ch, k=32, stride 16, 1 frame in
    let x = Tensor::randn(0f32, 1., (1, 512, 1), &d)?;
    let w = Tensor::randn(0f32, 1., (512, 1, 32), &d)?;
    bench("convtr dw 512 k32 s16 (upsample)", || {
        x.conv_transpose1d(&w, 0, 0, 16, 1, 512)
    })?;
    // seanet decoder stages: convtr C->C/2, k=2r, stride r
    for (cin, t, r) in [(512usize, 16usize, 6usize), (256, 96, 5), (128, 480, 4)] {
        let x = Tensor::randn(0f32, 1., (1, cin, t), &d)?;
        let w = Tensor::randn(0f32, 1., (cin, cin / 2, 2 * r), &d)?;
        bench(
            &format!("convtr {cin}->{} k{} s{r} T{t}", cin / 2, 2 * r),
            || x.conv_transpose1d(&w, 0, 0, r, 1, 1),
        )?;
        let c = cin / 2;
        let tt = t * r;
        let x = Tensor::randn(0f32, 1., (1, c, tt + 2), &d)?;
        let w = Tensor::randn(0f32, 1., (c / 2, c, 3), &d)?;
        bench(&format!("conv {c}->{} k3 T{tt}", c / 2), || {
            x.conv1d(&w, 0, 1, 1, 1)
        })?;
        let w2 = w.reshape((c / 2, c * 3))?;
        let a = x.conv1d(&w, 0, 1, 1, 1)?;
        let b = conv_mm(&x, &w2, 3, 1)?;
        let diff = (a - b)?.abs()?.max_all()?.to_scalar::<f32>()?;
        bench(
            &format!("im2col {c}->{} k3 T{tt} (diff {diff:.1e})", c / 2),
            || conv_mm(&x, &w2, 3, 1),
        )?;
    }
    let x = Tensor::randn(0f32, 1., (1, 64, 1920 + 6), &d)?;
    let w = Tensor::randn(0f32, 1., (1, 64, 7), &d)?;
    bench("conv 64->1 k7 T1920 (final)", || x.conv1d(&w, 0, 1, 1, 1))?;
    let x = Tensor::randn(0f32, 1., (1, 512, 16 + 6), &d)?;
    let w = Tensor::randn(0f32, 1., (512, 512, 7), &d)?;
    bench("conv 512->512 k7 T16 (first)", || x.conv1d(&w, 0, 1, 1, 1))?;
    Ok(())
}
