use crate::ModelState;
use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};
use std::collections::HashMap;

/// `w @ x[i]` for each batch item: `w` `(M, K)`, `x` `(B, K, N)` -> `(B, M, N)`.
/// (candle's CPU matmul mishandles a stride-0 batch from `broadcast_left`.)
fn batched_left_matmul(w: &Tensor, x: &Tensor) -> Result<Tensor> {
    let b = x.dim(0)?;
    if b == 1 {
        return w.matmul(&x.squeeze(0)?.contiguous()?)?.unsqueeze(0);
    }
    w.broadcast_left(b)?.contiguous()?.matmul(&x.contiguous()?)
}

#[derive(Clone)]
pub struct StreamingConv1d {
    conv: Conv1d,
    /// Weight as `[out, in * k]` for the im2col path (stride 1, one group).
    im2col_weight: Option<Tensor>,
    padding_mode: String,
    stride: usize,
    kernel_size: usize,
    dilation: usize,
    in_channels: usize,
    name: String,
}

impl StreamingConv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        padding_mode: &str,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = Conv1dConfig {
            stride,
            padding: 0,
            dilation,
            groups,
            ..Default::default()
        };
        let conv = if bias {
            candle_nn::conv1d(
                in_channels,
                out_channels,
                kernel_size,
                config,
                vb.pp("conv"),
            )?
        } else {
            candle_nn::conv1d_no_bias(
                in_channels,
                out_channels,
                kernel_size,
                config,
                vb.pp("conv"),
            )?
        };

        let im2col_weight = if stride == 1 && groups == 1 {
            Some(
                conv.weight()
                    .reshape((out_channels, in_channels * kernel_size))?
                    .contiguous()?,
            )
        } else {
            None
        };
        Ok(Self {
            conv,
            im2col_weight,
            padding_mode: padding_mode.to_string(),
            stride,
            kernel_size,
            dilation,
            in_channels,
            name: name.to_string(),
        })
    }

    /// Unpadded convolution. Stride-1 convs run as im2col + one matmul,
    /// about twice as fast as candle's CPU conv1d at the decoder's sizes.
    fn conv_forward(&self, x: &Tensor) -> Result<Tensor> {
        let Some(w2) = &self.im2col_weight else {
            return self.conv.forward(x);
        };
        let (b, c, t) = x.dims3()?;
        let k = self.kernel_size;
        let t_out = t + 1 - self.effective_kernel_size();
        let cols = if k == 1 {
            x.clone()
        } else {
            // cols[b, c*k + j, t] = x[b, c, t + j*dilation]
            let taps = (0..k)
                .map(|j| x.narrow(2, j * self.dilation, t_out))
                .collect::<Result<Vec<_>>>()?;
            Tensor::stack(&taps, 2)?.reshape((b, c * k, t_out))?
        };
        let y = batched_left_matmul(w2, &cols)?;
        match self.conv.bias() {
            Some(bias) => y.broadcast_add(&bias.reshape((1, (), 1))?),
            None => Ok(y),
        }
    }

    pub fn effective_kernel_size(&self) -> usize {
        (self.kernel_size - 1) * self.dilation + 1
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        _sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        let kernel = self.effective_kernel_size();
        let mut state = HashMap::new();
        if kernel > self.stride {
            let previous = Tensor::zeros(
                (batch_size, self.in_channels, kernel - self.stride),
                DType::F32,
                device,
            )?;
            state.insert("previous".to_string(), previous);
        }
        Ok(state)
    }

    pub fn forward(&self, x: &Tensor, model_state: &mut ModelState, step: usize) -> Result<Tensor> {
        let (b, c, t) = x.dims3()?;
        let s = self.stride;
        if t == 0 || t % s != 0 {
            return Err(candle_core::Error::Msg(format!(
                "Steps must be multiple of stride {}, got {}",
                s, t
            )));
        }

        // Auto-initialize state if missing
        if !model_state.contains_key(&self.name) {
            let init = self.init_state(b, t, x.device())?;
            model_state.insert(self.name.clone(), init);
        }

        let module_state = model_state.get_mut(&self.name).unwrap();
        let kernel = self.effective_kernel_size();
        let pad_left = kernel.saturating_sub(s);

        if pad_left > 0 {
            let previous = module_state
                .remove("previous")
                .ok_or_else(|| candle_core::Error::Msg("previous state not found".to_string()))?;
            let is_first = step == 0;

            let x_with_padding = if is_first && self.padding_mode == "replicate" {
                // Replicate the first frame for the initial padding
                let first_frame = x.narrow(2, 0, 1)?;
                let replicated_padding = first_frame.broadcast_as((b, c, pad_left))?;
                Tensor::cat(&[replicated_padding, x.clone()], 2)?
            } else {
                Tensor::cat(&[previous, x.clone()], 2)?
            };

            let y = self.conv_forward(&x_with_padding)?;

            // Update previous state for next call
            let total_len = x_with_padding.dims()[2];
            let new_previous = x_with_padding.narrow(2, total_len - pad_left, pad_left)?;
            module_state.insert("previous".to_string(), new_previous);

            Ok(y)
        } else {
            self.conv_forward(x)
        }
    }

    pub fn weight(&self) -> &Tensor {
        self.conv.weight()
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.conv.bias()
    }
}

#[derive(Clone)]
pub struct StreamingConvTranspose1d {
    convtr: ConvTranspose1d,
    /// Depthwise (groups == channels) with kernel a multiple of the stride:
    /// computed by `depthwise_forward` instead of the generic grouped kernel,
    /// which is two orders of magnitude slower in candle 0.11.
    depthwise: bool,
    /// Single-group weight as `[out * k, in]` (kernel a multiple of the
    /// stride) for `matmul_forward`.
    matmul_weight: Option<Tensor>,
    stride: usize,
    kernel_size: usize,
    out_channels: usize,
    name: String,
}

impl StreamingConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = ConvTranspose1dConfig {
            stride,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups,
        };
        let convtr = if bias {
            candle_nn::conv_transpose1d(
                in_channels,
                out_channels,
                kernel_size,
                config,
                vb.pp("convtr"),
            )?
        } else {
            candle_nn::conv_transpose1d_no_bias(
                in_channels,
                out_channels,
                kernel_size,
                config,
                vb.pp("convtr"),
            )?
        };

        let depthwise = groups > 1
            && groups == in_channels
            && groups == out_channels
            && kernel_size.is_multiple_of(stride);
        let matmul_weight = if groups == 1 && kernel_size.is_multiple_of(stride) {
            // [in, out, k] -> [out, k, in] -> [out * k, in]
            Some(
                convtr
                    .weight()
                    .permute((1, 2, 0))?
                    .reshape((out_channels * kernel_size, in_channels))?
                    .contiguous()?,
            )
        } else {
            None
        };
        Ok(Self {
            convtr,
            depthwise,
            matmul_weight,
            stride,
            kernel_size,
            out_channels,
            name: name.to_string(),
        })
    }

    /// Depthwise transposed conv: `y[c, t*s + j] += x[c, t] * w[c, j]`.
    /// With `k = m*s`, block `i` of each kernel lands at offset `i*s`.
    fn depthwise_forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t) = x.dims3()?;
        let k = self.kernel_size;
        let w = self.convtr.weight().reshape((1, c, k, 1))?;
        // (B, C, K, T)
        let z = x.unsqueeze(2)?.broadcast_mul(&w)?;
        self.overlap_add(&z, b, c, t)
    }

    /// Transposed conv as one matmul plus overlap-add:
    /// `z[o, j, t] = sum_c w[c, o, j] x[c, t]`, then `y[o, t*s + j] += z[o, j, t]`.
    fn matmul_forward(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let (b, _c, t) = x.dims3()?;
        let k = self.kernel_size;
        let out = self.out_channels;
        // (B, out, k, T)
        let z = batched_left_matmul(w, x)?.reshape((b, out, k, t))?;
        self.overlap_add(&z, b, out, t)
    }

    /// Sums `z` `(B, C, k, T)` into `(B, C, (T - 1) * s + k)`, block `i` of
    /// each kernel landing at offset `i * s`, then adds the bias.
    fn overlap_add(&self, z: &Tensor, b: usize, c: usize, t: usize) -> Result<Tensor> {
        let (k, s) = (self.kernel_size, self.stride);
        let m = k / s;
        let mut y: Option<Tensor> = None;
        for i in 0..m {
            // (B, C, s, T) -> (B, C, T, s) -> (B, C, T * s)
            let block = z
                .narrow(2, i * s, s)?
                .transpose(2, 3)?
                .reshape((b, c, t * s))?
                .pad_with_zeros(2, i * s, (m - 1 - i) * s)?;
            y = Some(match y {
                Some(acc) => (acc + block)?,
                None => block,
            });
        }
        let y = y.expect("kernel_size >= stride");
        match self.convtr.bias() {
            Some(bias) => y.broadcast_add(&bias.reshape((1, c, 1))?),
            None => Ok(y),
        }
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        _sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        let mut state = HashMap::new();
        let k = self.kernel_size;
        let s = self.stride;
        if k > s {
            let partial =
                Tensor::zeros((batch_size, self.out_channels, k - s), DType::F32, device)?;
            state.insert("partial".to_string(), partial);
        }
        Ok(state)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        model_state: &mut ModelState,
        _step: usize,
    ) -> Result<Tensor> {
        let (b, _c, t) = x.dims3()?;
        let k = self.kernel_size;
        let s = self.stride;
        let trim = k.saturating_sub(s);

        // Auto-initialize state if missing
        if !model_state.contains_key(&self.name) {
            let init = self.init_state(b, t, x.device())?;
            model_state.insert(self.name.clone(), init);
        }

        let module_state = model_state.get_mut(&self.name).unwrap();

        let mut y = if self.depthwise {
            self.depthwise_forward(x)?
        } else if let Some(w) = &self.matmul_weight {
            self.matmul_forward(x, w)?
        } else {
            self.convtr.forward(x)?
        };

        if trim > 0 {
            if let Some(partial) = module_state.remove("partial") {
                // y is (B, C, S*T + trim)
                // We add partial to the start of y
                let y_head = y.narrow(2, 0, trim)?;
                let y_sum = (y_head + partial)?;
                let y_tail = y.narrow(2, trim, y.dims()[2] - trim)?;
                y = Tensor::cat(&[y_sum, y_tail], 2)?;
            }

            // The last `trim` elements of `y` become the next `partial`
            let len = y.dims()[2];
            let mut next_partial = y.narrow(2, len - trim, trim)?;

            // If bias exists, we need to subtract it from the partial state
            // because it will be added again when we run the next forward pass.
            if let Some(bias) = self.convtr.bias() {
                let b_reshaped = bias.reshape((self.out_channels, 1))?;
                next_partial = next_partial.broadcast_sub(&b_reshaped)?;
            }
            module_state.insert("partial".to_string(), next_partial);

            // The output we actually return is y MINUS the new partial tail
            y = y.narrow(2, 0, len - trim)?;
        }

        Ok(y)
    }

    pub fn weight(&self) -> &Tensor {
        self.convtr.weight()
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.convtr.bias()
    }
}

#[derive(Clone)]
pub struct ConvDownsample1d {
    conv: StreamingConv1d,
}

impl ConvDownsample1d {
    pub fn new(
        stride: usize,
        dimension: usize,
        out_dimension: usize,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = StreamingConv1d::new(
            dimension,
            out_dimension,
            2 * stride,
            stride,
            1,
            1,
            false,
            "replicate",
            &format!("{}.conv", name),
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        self.conv.init_state(batch_size, sequence_length, device)
    }

    pub fn forward(&self, x: &Tensor, model_state: &mut ModelState, step: usize) -> Result<Tensor> {
        self.conv.forward(x, model_state, step)
    }
}

#[derive(Clone)]
pub struct ConvTrUpsample1d {
    convtr: StreamingConvTranspose1d,
}

impl ConvTrUpsample1d {
    pub fn new(
        stride: usize,
        dimension: usize,
        in_dimension: usize,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let convtr = StreamingConvTranspose1d::new(
            in_dimension,
            dimension,
            2 * stride,
            stride,
            dimension,
            false,
            &format!("{}.convtr", name),
            vb.pp("convtr"),
        )?;
        Ok(Self { convtr })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        self.convtr.init_state(batch_size, sequence_length, device)
    }

    pub fn forward(&self, x: &Tensor, model_state: &mut ModelState, step: usize) -> Result<Tensor> {
        self.convtr.forward(x, model_state, step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depthwise_convtr_matches_grouped_kernel() -> Result<()> {
        let dev = candle_core::Device::Cpu;
        let (c, k, stride) = (8, 6, 3);
        let mut map = HashMap::new();
        map.insert(
            "convtr.weight".to_string(),
            Tensor::randn(0f32, 1.0, (c, 1, k), &dev)?,
        );
        let vb = VarBuilder::from_tensors(map, DType::F32, &dev);
        let m = StreamingConvTranspose1d::new(c, c, k, stride, c, false, "t", vb)?;
        assert!(m.depthwise);
        let x = Tensor::randn(0f32, 1.0, (2, c, 5), &dev)?;
        let fast = m.depthwise_forward(&x)?;
        let slow = m.convtr.forward(&x)?;
        let diff = (fast - slow)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "max diff {diff}");
        Ok(())
    }

    fn vb_with(name: &str, t: Tensor) -> VarBuilder<'static> {
        let mut map = HashMap::new();
        let dev = t.device().clone();
        map.insert(name.to_string(), t);
        VarBuilder::from_tensors(map, DType::F32, &dev)
    }

    #[test]
    fn im2col_conv_matches_candle() -> Result<()> {
        let dev = candle_core::Device::Cpu;
        for (k, dil) in [(1, 1), (3, 1), (7, 1), (3, 2)] {
            let w = Tensor::randn(0f32, 1.0, (6, 5, k), &dev)?;
            let m = StreamingConv1d::new(
                5,
                6,
                k,
                1,
                dil,
                1,
                false,
                "constant",
                "t",
                vb_with("conv.weight", w),
            )?;
            let x = Tensor::randn(0f32, 1.0, (2, 5, 20), &dev)?;
            let diff = (m.conv_forward(&x)? - m.conv.forward(&x)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(diff < 1e-4, "k={k} dil={dil}: {diff}");
        }
        Ok(())
    }

    /// Checked against a hand-written loop: candle 0.11's conv_transpose1d
    /// returns wrong values for every batch item after the first.
    #[test]
    fn matmul_convtr_matches_reference() -> Result<()> {
        let dev = candle_core::Device::Cpu;
        let (cin, cout, k, s, t) = (6, 4, 8, 4, 7);
        let w = Tensor::randn(0f32, 1.0, (cin, cout, k), &dev)?;
        let m = StreamingConvTranspose1d::new(
            cin,
            cout,
            k,
            s,
            1,
            false,
            "t",
            vb_with("convtr.weight", w),
        )?;
        let x = Tensor::randn(0f32, 1.0, (2, cin, t), &dev)?;
        let wm = m.matmul_weight.clone().unwrap();
        let ours = m.matmul_forward(&x, &wm)?;
        let w3 = m.convtr.weight().to_vec3::<f32>()?;
        for bi in 0..2 {
            let xv = x.get(bi)?.to_vec2::<f32>()?;
            let mut y = vec![vec![0f32; (t - 1) * s + k]; cout];
            for c in 0..cin {
                for o in 0..cout {
                    for ti in 0..t {
                        for j in 0..k {
                            y[o][ti * s + j] += w3[c][o][j] * xv[c][ti];
                        }
                    }
                }
            }
            let reference = Tensor::new(y, &dev)?;
            let diff = (ours.get(bi)? - reference)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(diff < 1e-4, "batch {bi}: {diff}");
        }
        Ok(())
    }
}
