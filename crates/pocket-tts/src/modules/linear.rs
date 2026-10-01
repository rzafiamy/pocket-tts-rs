//! Linear layers that run either dense (f32) or quantized (GGUF `QMatMul`).
//!
//! Model constructors call [`linear`] / [`linear_no_bias`] with a regular
//! `VarBuilder`. While a GGUF model is being built, [`QuantScope`] registers
//! its quantized weights; a constructor whose `<prefix>.weight` is registered
//! gets a `QMatMul` instead of a dense matrix. Batch-1 decoding is bound by
//! weight bandwidth, so this is where quantization pays.

use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

pub type QuantWeights = HashMap<String, Arc<QTensor>>;

thread_local! {
    static QUANT: RefCell<Option<Arc<QuantWeights>>> = const { RefCell::new(None) };
}

/// Makes `weights` visible to the linear constructors of this thread until dropped.
pub struct QuantScope;

impl QuantScope {
    pub fn enter(weights: Arc<QuantWeights>) -> Self {
        QUANT.with(|q| *q.borrow_mut() = Some(weights));
        QuantScope
    }
}

impl Drop for QuantScope {
    fn drop(&mut self) {
        QUANT.with(|q| *q.borrow_mut() = None);
    }
}

fn registered(name: &str) -> Option<Arc<QTensor>> {
    QUANT.with(|q| q.borrow().as_ref().and_then(|m| m.get(name).cloned()))
}

#[derive(Clone, Debug)]
pub enum Linear {
    Dense(candle_nn::Linear),
    Quant {
        weight: QMatMul,
        bias: Option<Tensor>,
        out_features: usize,
    },
}

impl Linear {
    pub fn out_features(&self) -> Result<usize> {
        match self {
            Linear::Dense(l) => l.weight().dim(0),
            Linear::Quant { out_features, .. } => Ok(*out_features),
        }
    }
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Linear::Dense(l) => l.forward(xs),
            Linear::Quant { weight, bias, .. } => {
                let ys = weight.forward(xs)?;
                match bias {
                    Some(b) => ys.broadcast_add(b),
                    None => Ok(ys),
                }
            }
        }
    }
}

fn build(in_dim: usize, out_dim: usize, with_bias: bool, vb: VarBuilder) -> Result<Linear> {
    if let Some(q) = registered(&format!("{}.weight", vb.prefix())) {
        let dims = q.shape().dims();
        if dims != [out_dim, in_dim] {
            candle_core::bail!(
                "{}.weight: expected [{out_dim}, {in_dim}], got {dims:?}",
                vb.prefix()
            );
        }
        let bias = if with_bias {
            Some(vb.get(out_dim, "bias")?)
        } else {
            None
        };
        return Ok(Linear::Quant {
            weight: QMatMul::from_arc(q)?,
            bias,
            out_features: out_dim,
        });
    }
    let l = if with_bias {
        candle_nn::linear(in_dim, out_dim, vb)?
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)?
    };
    Ok(Linear::Dense(l))
}

pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    build(in_dim, out_dim, true, vb)
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    build(in_dim, out_dim, false, vb)
}
