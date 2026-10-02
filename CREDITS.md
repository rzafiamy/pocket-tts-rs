# Credits

pocket-tts-rs is distributed under MIT. It builds on the work
below, each used under its own license. Versions are those of `Cargo.lock`.

## Authors

- Kyutai — Pocket TTS model, training and Python reference
  ([kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts), MIT;
  weights CC-BY-4.0 on Hugging Face `kyutai/pocket-tts`).
- [@babybirdprd](https://github.com/babybirdprd) and contributors — the
  original Candle port this fork starts from
  ([babybirdprd/pocket-tts](https://github.com/babybirdprd/pocket-tts), MIT).
- Rija Z. ([@rzafiamy](https://github.com/rzafiamy)) — this fork: upstream
  catch-up, GGUF, quantization, CPU/GPU optimization.
- Developed with the assistance of Claude (Anthropic) through Claude Code.

## Third-party sources

- Model weights, tokenizers and predefined voices are downloaded from
  Hugging Face (`kyutai/pocket-tts`, gated; `kyutai/pocket-tts-without-voice-cloning`).
  CC-BY-4.0: credit Kyutai when distributing generated GGUF files.
- The configs in `crates/pocket-tts/config/` are copied from upstream (MIT).
- `assets/ref.wav` and the reference tensors come from the original Candle port.

## Rust dependencies

| Dependency | Version | Role | License | Source |
|---|---|---|---|---|
| `candle-core` | 0.11.0 | Tensors, quantized matmul (CPU/CUDA/Metal), GGUF read/write | MIT OR Apache-2.0 | https://github.com/huggingface/candle |
| `candle-nn` | 0.11.0 | Layers, softmax, VarBuilder | MIT OR Apache-2.0 | https://github.com/huggingface/candle |
| `tokenizers` | 0.21.4 | SentencePiece / `tokenizer.json` | Apache-2.0 | https://github.com/huggingface/tokenizers |
| `hf-hub` | 0.4.3 | Hugging Face downloads, cached token | Apache-2.0 | https://github.com/huggingface/hf-hub |
| `safetensors` | 0.5.3 | Upstream weights and voices | Apache-2.0 | https://github.com/huggingface/safetensors |
| `serde` | 1.0.228 | Serialization | MIT OR Apache-2.0 | https://serde.rs |
| `serde_json` | 1.0.149 | JSON API | MIT OR Apache-2.0 | https://github.com/serde-rs/json |
| `serde_yaml` | 0.9.34 | Model configs | MIT OR Apache-2.0 | https://github.com/dtolnay/serde-yaml |
| `anyhow` | 1.0.100 | Errors | MIT OR Apache-2.0 | https://github.com/dtolnay/anyhow |
| `thiserror` | 2.0.18 | Error types | MIT OR Apache-2.0 | https://github.com/dtolnay/thiserror |
| `hound` | 3.5.1 | WAV I/O | Apache-2.0 | https://github.com/ruuda/hound |
| `rubato` | 0.14.1 | Resampling of voice prompts | MIT | https://github.com/HEnquist/rubato |
| `rand` | 0.8.5 | Sampling noise | MIT OR Apache-2.0 | https://github.com/rust-random/rand |
| `rand_distr` | 0.4.3 | Truncated normal noise | MIT OR Apache-2.0 | https://github.com/rust-random/rand |
| `rayon` | 1.11.0 | CPU parallelism | MIT OR Apache-2.0 | https://github.com/rayon-rs/rayon |
| `regex` | 1.12.2 | Pause markers | MIT OR Apache-2.0 | https://github.com/rust-lang/regex |
| `memmap2` | 0.9.9 | Memory-mapped weights | MIT OR Apache-2.0 | https://github.com/RazrFalcon/memmap2-rs |
| `byteorder` | 1.5.0 | Binary parsing | MIT OR Unlicense | https://github.com/BurntSushi/byteorder |
| `lenient_semver` | 0.4.2 | Version parsing | MIT OR Apache-2.0 | https://github.com/knutwalker/lenient-semver |
| `tracing` | 0.1.44 | Logging | MIT | https://github.com/tokio-rs/tracing |
| `tracing-subscriber` | 0.3.22 | Log output | MIT | https://github.com/tokio-rs/tracing |
| `clap` | 4.5.54 | Command line | MIT OR Apache-2.0 | https://github.com/clap-rs/clap |
| `axum` | 0.7.9 | HTTP server | MIT | https://github.com/tokio-rs/axum |
| `tokio` | 1.49.0 | Async runtime | MIT | https://github.com/tokio-rs/tokio |
| `tokio-stream` | 0.1.18 | Streaming responses | MIT | https://github.com/tokio-rs/tokio |
| `futures-util` | 0.3.31 | Stream helpers | MIT OR Apache-2.0 | https://github.com/rust-lang/futures-rs |
| `tower` | 0.4.13 | Router calls in tests | MIT | https://github.com/tower-rs/tower |
| `tower-http` | 0.5.2 | CORS, tracing middleware | MIT | https://github.com/tower-rs/tower-http |
| `base64` | 0.21.7 | Base64 voice uploads | MIT OR Apache-2.0 | https://github.com/marshallpierce/rust-base64 |
| `indicatif` | 0.17.11 | Progress bar | MIT | https://github.com/console-rs/indicatif |
| `owo-colors` | 4.2.3 | Terminal colors | MIT | https://github.com/jam1garner/owo-colors |
| `intel-mkl-src` | 0.8.1 | Optional MKL backend (`--features mkl`) | MIT OR Apache-2.0 (MKL: Intel Simplified Software License) | https://github.com/rust-math/intel-mkl-src |
| `criterion` | 0.8.1 | Benchmarks (dev) | MIT OR Apache-2.0 | https://github.com/bheisler/criterion.rs |
| `assert_cmd` | 2.1.2 | CLI tests (dev) | MIT OR Apache-2.0 | https://github.com/assert-rs/assert_cmd |

No Tauri library: pocket-tts-rs is a CLI and an HTTP server. No bundled
assets (icons, fonts).
