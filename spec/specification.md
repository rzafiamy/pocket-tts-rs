# Specification

## Goal

Run Kyutai's Pocket TTS (English and French first, the other upstream
languages through the same code) as a single native binary and a single
GGUF model file, faster and lighter than the Python reference **without
degrading quality**, and serve it over an OpenAI-compatible HTTP API so that
zallama can host it.

## Context and users

Pocket TTS is a ~100M-parameter streaming TTS (FlowLM backbone + sampler
head + Mimi codec) released by Kyutai with a PyTorch implementation. A
community Candle port existed but stopped before the multilingual models,
had no real quantization and ran at half the speed of PyTorch.

Users:
- **zallama integrator**: declares the server as a TTS backend; needs
  `/health`, `/v1/audio/speech`, a known VRAM footprint and one model file.
- **Application developer**: sends text, gets WAV or streamed PCM within
  tens of milliseconds of first audio.
- **Command-line user**: synthesizes a sentence or a file, optionally in a
  cloned voice.
- **Maintainer**: follows upstream releases, converts new checkpoints and
  checks parity and quality.

Main path: `pocket-tts convert --variant french` → `pocket-tts serve -m
french-q8_0.gguf` (zallama) or `pocket-tts generate -m ...`.

## Priorities

- **Must (core, MVP)**: English and French models, parity with upstream,
  GGUF single file, quantized inference without quality loss, CPU and CUDA,
  HTTP API.
- **Should**: voice cloning from WAV, explicit pauses, all upstream
  variants (24-layer, drifting, other languages).
- **Could / optional**: Metal, more languages tested end to end.

## Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| REQ-CFG-001 | Every upstream config (languages, `_24l`, drifting, `b6369a24`) is built into the binary and selectable with `--variant`. | Must |
| REQ-TXT-001 | Text preparation (replacements, punctuation, capitalization, padding, chunking into ≤ 50-token sentence groups) is identical to upstream `text_chunking.py`. | Must |
| REQ-TXT-002 | `[pause:500ms]` / `[pause:1s]` markers insert exact silence; punctuation is left to the model. | Should |
| REQ-VOI-001 | Predefined voices (upstream exported model states) load for each language; each language has a native default voice. | Must |
| REQ-VOI-002 | A WAV file clones a voice (Mimi encoder, end-on-pause trimming, `bos_before_voice`). | Should |
| REQ-INF-001 | At temperature 0 the output matches the Python reference: same length, correlation ≥ 0.999 on short prompts. | Must |
| REQ-INF-002 | Generation streams audio frame by frame and stops on EOS like upstream (EOS ignored for 6 frames, same tail). | Must |
| REQ-GGF-001 | `convert` writes one GGUF holding weights, config, tokenizer and voices; `--model` needs no other file. | Must |
| REQ-GGF-002 | An f32 GGUF reproduces the safetensors model. | Must |
| REQ-GGF-003 | Quantized GGUF (q8_0, q4k) linear layers run quantized; q8_0 is < 180 MB. | Must |
| REQ-QUA-001 | Quantization does not reduce intelligibility: ASR word error rate within noise of f32 on the EN/FR sets. | Must |
| REQ-OPS-001 | Optimized kernels (im2col conv, matmul transposed conv, exp GELU, copy-free attention) equal reference implementations. | Must |
| REQ-PRF-001 | CPU, q8_0: ≥ 5× real time and < 100 ms to first audio on a desktop CPU. | Must |
| REQ-GPU-001 | CUDA: ≥ 20× real time, < 1 GiB VRAM, output equal to CPU/Python at temperature 0. | Must |
| REQ-SRV-001 | HTTP server: `/health`, `/generate`, `/stream`, `/tts`, OpenAI `/v1/audio/speech`. | Must |
| REQ-CLI-001 | `generate` writes a WAV (or PCM to stdout with `--stream`). | Must |

## Technical specification

### Architecture

Two crates in one Cargo workspace:

- `crates/pocket-tts` (library): model loading, text preparation,
  generation loop, kernels.
- `crates/pocket-tts-cli` (binary `pocket-tts`): commands `generate`,
  `convert`, `info`, `serve`; shared model options in `loader.rs`; HTTP
  server (axum) in `server/`.

Pipeline per request: text → `text_chunking.rs` (≤ 50-token chunks) →
FlowLM transformer (`models/flow_lm.rs`, `models/transformer.rs`) with the
voice as a prefilled KV cache → sampler head (`modules/mlp.rs`: lsd,
flow_matching or drifting) → one 32-dim latent per 80 ms frame → Mimi
decoder (`models/mimi.rs`, `models/seanet.rs`) → 24 kHz PCM, streamed frame
by frame.

### Modules

| Module | Role |
|---|---|
| `tts_model.rs` | Load (safetensors or GGUF), voice prompts, generation loop |
| `builtin_configs.rs`, `config.rs` | Upstream YAML configs compiled in |
| `gguf.rs`, `modules/linear.rs` | GGUF read/write, dense or `QMatMul` linears |
| `modules/conv.rs`, `sdpa.rs`, `activations.rs` | CPU-critical kernels (im2col conv, matmul transposed conv, attention, GELU) |
| `voices.rs`, `voice_state.rs` | Predefined voices, attention-state init and cloning |
| `text_chunking.rs`, `pause.rs` | Text preparation, `[pause:…]` markers |

### Data

- **GGUF model file**: tensors (f32/f16/q8_0/q6k/q5k/q4k/q4_0), the model
  config and tokenizer as metadata, voices as tensors. Self-sufficient.
- **Hugging Face cache**: safetensors weights, tokenizer and voices when no
  GGUF is given (`hf://` paths, `HF_HOME`).
- No database, no configuration file; options and `POCKET_TTS_*`
  environment variables only.

### APIs and commands

- CLI: `pocket-tts generate | convert | info | serve` (`--help` on each).
- HTTP: `GET /health`, `POST /generate`, `POST /stream` (PCM),
  `POST /tts` (multipart), `POST /v1/audio/speech` (OpenAI); details in
  `docs/serve.md`.
- Rust: `TTSModel::load`, `load_gguf`, `generate`, `generate_stream`
  (`docs/rust-api.md`).

### Platforms

Linux (x86_64, aarch64), Windows (x86_64) and macOS (aarch64, x86_64).
Backends selected at build time by Cargo features: CPU (default), `cuda`,
`metal`, `mkl`; at run time by `--device`. Release binaries are built by
`.github/workflows/release.yml` with portable CPU targets (`x86-64-v3`,
`apple-m1`); local builds use `target-cpu=native`.

## Rules

- Upstream is the reference: when the Python code changes behavior, the port
  follows (see `docs/porting-status.md`). Deviations are documented.
- No speed optimization ships without a test against a reference
  implementation (REQ-OPS-001) and the parity matrix (REQ-INF-001).
- Model weights stay under Kyutai's license (CC-BY-4.0); voice-cloning
  weights are gated on Hugging Face and need an accepted license + token.

## Known limitations

- Exact bit parity is not possible: threaded float reductions differ, and
  the autoregressive loop amplifies the difference on long chunks (corr
  0.92 on a 3-chunk French text, same length and EOS).
- Pinning the process to few CPUs (`taskset`) is very slow (see
  `docs/performance.md`).
- Metal builds compile but are not measured; CUDA on Windows is untested.
- Only English and French were evaluated for quality; other languages pass
  the parity matrix only for German.
