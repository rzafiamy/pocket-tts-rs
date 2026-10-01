# Performance and quality

Measured on 2026-10-01/02, i9-14900K (8 P-cores + 16 E-cores), RTX 4090.
Text: three French sentences (~11 s of audio), default voice, temperature 0.3.
Tools: `cargo run --release --example bench -- <variant|file.gguf>` (end to end)
and `--example profile` (per-component time per 80 ms frame).

## CPU

| build | FlowLM ms/frame | Mimi ms/frame | real-time factor |
|---|---|---|---|
| fork as found (candle 0.9, f32) | 21 | 18 | 2.0x |
| Python reference (torch 2.14 CPU, 1 thread) | | | 4.4x |
| this port, f32 (safetensors or GGUF) | 20 | 7.2 | 2.8x |
| this port, GGUF q8_0 | 3.4 | 7.2 | **6.7–7.5x** (first audio 34–47 ms) |
| this port, GGUF q4k | 2.4 | 7.0 | **7–9x** (first audio 27–40 ms) |

The gains, in order of size:

1. **Quantized linear layers** (`QMatMul`). Batch-1 decoding reads every
   weight once per frame; f32 weights do not fit in cache and the 6-layer
   backbone is bandwidth bound. Q8_0 cuts FlowLM from 20 to 3.4 ms.
2. **Convolutions as matrix products.** Stride-1 convs go through im2col +
   one matmul; transposed convs through one matmul + overlap-add. candle
   0.11's grouped `conv_transpose1d` (Mimi's depthwise upsampler) took
   161 ms per frame.
3. **GELU through `exp`**: `0.5 (1 + tanh z) = sigmoid(2z)`. libm `tanh` costs
   16 ns per element, which made GELU the most expensive op of Mimi's
   transformer.
4. **Attention without copies**: the decode step reads the KV cache in
   place; masks are built on the host instead of a dozen tensor ops.

## GPU (RTX 4090, `--features cuda`, `--device cuda`)

| file | real-time factor | first chunk | peak VRAM |
|---|---|---|---|
| GGUF q8_0 | 28x | 5 ms | 680 MiB |
| GGUF q4k | 28x | 5 ms | 648 MiB |
| GGUF f32 | 27x | 5 ms | 936 MiB |

A frame costs ~2.7 ms (FlowLM 1.4, Mimi 1.3), dominated by kernel launches,
so quantization changes the footprint, not the speed. A bare candle CUDA
context with cuBLAS already takes 428 MiB; the q8_0 model adds ~250 MiB
(weights, KV caches, activations). CUDA's lazy module loading is the
default and matters: `CUDA_MODULE_LOADING=EAGER` raises the peak to 924 MiB.
Output at temperature 0 matches the Python reference (corr 1.00000); WER on
the French and English sets is 3.4% and 3.2%.

## CPU threads

Every op is small at batch 1, so threads mostly add synchronization:
1 to 4 threads give the same speed, 16 threads lose ~2x, 32 lose more.
The CLI defaults to 4 (`--threads`, or `RAYON_NUM_THREADS`).

Do not pin the process to a few CPUs (`taskset`): some kernels keep their own
spinning thread pool sized to the machine, and pinned to 4 CPUs a frame takes
~1 s instead of 10 ms.

## Quality

Word error rate from Parakeet TDT 0.6B v3 (through zallama) on 10 English
and 10 French sentences, two generations each (`scripts/eval/quality.py`):

| system | French WER | English WER |
|---|---|---|
| Python reference | 2.5% | 3.2% |
| Rust, f32 | 4.7% | 3.2% |
| Rust, GGUF q8_0 | 4.7% | 3.2% |
| Rust, GGUF q4k | 4.2% | 2.8% |

Most counted errors are normalization ("riverbank", "12%", "pourcent");
the French gap is mostly "chat gris" heard as "chagri". Quantizing the linear
layers to q8_0 or q4k does not change intelligibility.

Exact parity with Python is checked at temperature 0 by
`scripts/parity/matrix.sh` (see porting-status.md).

## Model files

| file | size |
|---|---|
| upstream safetensors (bf16) | 219 MB |
| GGUF f32 | 442 MB |
| GGUF q8_0 | 155 MB |
| GGUF q4k | 106 MB |

GGUF files embed one voice (the language default) unless `--voices` says
otherwise; each voice adds ~3.8 MB.
