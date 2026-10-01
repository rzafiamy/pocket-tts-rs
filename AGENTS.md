# AGENTS.md

Guidance for AI agents working in this repository.

## Project

Rust/Candle port of Kyutai's Pocket TTS, tracking upstream
kyutai-labs/pocket-tts (commit in `docs/porting-status.md`). One binary
(`pocket-tts`), models from built-in configs or single-file GGUF.

Pipeline: text → `text_chunking.rs` (upstream rules, ≤ 50-token chunks) →
FlowLM backbone (`models/flow_lm.rs`, `models/transformer.rs`) → sampler
head (`modules/mlp.rs`: lsd / flow_matching / drifting) → 32-dim latent per
80 ms frame → Mimi decoder (`models/mimi.rs`, `models/seanet.rs`) → 24 kHz.
Voices are FlowLM attention states (`voice_state.rs`, `voices.rs`).

## Where things are

- `crates/pocket-tts/src/tts_model.rs`: loading (safetensors / GGUF), voice
  prompts, generation loop.
- `src/gguf.rs`: GGUF format; `src/modules/linear.rs`: dense / QMatMul linears.
- `src/modules/conv.rs`, `sdpa.rs`, `activations.rs`: performance-critical
  kernels, each tested against a reference.
- `crates/pocket-tts-cli/src/`: CLI (`commands/`), shared model options
  (`loader.rs`), HTTP server (`server/`).

## Commands

```bash
cargo build --release -p pocket-tts-cli
cargo test --release --workspace
cargo run --release --example profile -- models/french-q8_0.gguf   # ms per frame
cargo run --release --example bench -- french                       # real-time factor
scripts/parity/matrix.sh                                            # vs Python
```

CUDA: `--features cuda`, build into `--target-dir target-cuda`, needs
`/usr/local/cuda/bin` in PATH and `CUDA_COMPUTE_CAP`.

## Rules

- Upstream behavior is the spec. Check a change against the Python
  reference at temperature 0 (same sample count, correlation ≥ 0.999 on
  short prompts).
- candle 0.11 pitfalls found here: grouped `conv_transpose1d` is very slow
  and wrong for batch > 1; `matmul` with a stride-0 batch (`broadcast_left`)
  returns wrong values; libm `tanh` is slow. Prefer the helpers in
  `modules/`.
- Batch-1 CPU decoding is not helped by many threads; measure with 1–4.
