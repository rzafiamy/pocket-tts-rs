# pocket-tts-rs

[Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts) in Rust
([Candle](https://github.com/huggingface/candle)): one native binary, one
GGUF model file, English and French speech (and the other upstream
languages) faster than the Python reference on CPU and 28× real time on GPU.

Forked from [babybirdprd/pocket-tts](https://github.com/babybirdprd/pocket-tts)
(Candle port, February 2026) and brought up to upstream `41cbc84`
(2026-10-01): multilingual models, GGUF, quantization, CPU and CUDA
optimizations. Overview with figures: [portfolio/](portfolio/README.md).

| | Python reference | pocket-tts-rs |
|---|---|---|
| CPU, French (i9-14900K) | 4.4× real time | 6.7–7.5× (q8_0), 7–9× (q4k), first audio ~40 ms |
| GPU (RTX 4090) | | 28× real time, first audio 5 ms, 680 MiB VRAM |
| Model files | safetensors + tokenizer + voices | one GGUF, 155 MB (q8_0) |

## Features

Core (must work, covered by tests — see [spec/matrix.md](spec/matrix.md)):

- **Upstream models**: English (2026-01/04/09, 24-layer, drifting sampler
  head), French, German, Italian, Spanish, Portuguese, Dutch. Configs are
  built in (`crates/pocket-tts/src/builtin_configs.rs`), selected with
  `--variant`.
- **Upstream text handling**: character replacements, punctuation,
  capitalization, chunks of ≤ 50 tokens on sentence boundaries
  (`crates/pocket-tts/src/text_chunking.rs`).
- **Voices**: 27 predefined voices per language with a native default
  (`estelle` for French, `alba` for English) — `crates/pocket-tts/src/voices.rs`.
- **GGUF**: `pocket-tts convert` writes weights, config, tokenizer and voices
  into one file; linear layers in q8_0/q4k run quantized
  (`crates/pocket-tts/src/gguf.rs`, `crates/pocket-tts/src/modules/linear.rs`).
- **Fast CPU path**: im2col convolutions, matmul transposed convolutions,
  exp-based GELU, copy-free attention (`crates/pocket-tts/src/modules/`).
- **CUDA** (`--features cuda`, `--device cuda`) and Metal (`--features metal`, untested).
- **Streaming**: audio comes out frame by frame (80 ms); `--stream` writes PCM to stdout.
- **HTTP server** with an OpenAI-compatible `/v1/audio/speech`
  (`crates/pocket-tts-cli/src/server/`).

Secondary (optional):

- **Voice cloning** from a WAV file (needs the gated weights, see below).
- **Explicit pauses**: `[pause:500ms]`, `[pause:1s]` (`crates/pocket-tts/src/pause.rs`).

## Installation

### Prerequisites

- Linux, macOS or Windows; a C/C++ compiler, `pkg-config`, OpenSSL headers, git, curl.
- Rust 1.97.1 (pinned in `rust-toolchain.toml`, installed by rustup).
- NVIDIA GPU: CUDA Toolkit ≥ 12 (`nvcc`). Apple GPU: Xcode.
- Optional: Python 3 for the parity and quality scripts.

`./prereq.sh` checks all of this and installs Rust and the build packages
when missing (`CHECK_ONLY=1 ./prereq.sh` only checks).

### Commands

```bash
./setup.sh            # prerequisites, cargo fetch, cargo check   (--cuda for GPU)
./build.sh            # release binary in build/pocket-tts-<os>-cpu-<version>
./build.sh --cuda     # build/pocket-tts-<os>-cuda-<version>
```

### Check the installation

```bash
build/pocket-tts-linux-cpu-0.7.0 --version
build/pocket-tts-linux-cpu-0.7.0 generate --variant french -t "Bonjour." -o bonjour.wav
tests/e2e.sh build/pocket-tts-linux-cpu-0.7.0     # convert + generate + serve
```

### Model access

Weights download from Hugging Face on first use and are cached in
`~/.cache/huggingface`. The voice-cloning weights (`kyutai/pocket-tts`) are
gated: accept the license on the model page, then `hf auth login` or set
`HF_TOKEN`. Without access, the ungated weights load instead: everything
works except cloning from a WAV.

## Usage

```bash
# Speak (built-in variant, downloads on first use)
pocket-tts generate --variant french -t "Bonjour le monde." -o out.wav
pocket-tts generate --variant english -v alba -t "Hello world." -o out.wav

# Clone a voice from a WAV file
pocket-tts generate --variant french -v my_voice.wav -t "Ceci est ma voix." -o out.wav

# Convert once to a single GGUF, then use it
pocket-tts convert --variant french --dtype q8_0          # -> french-q8_0.gguf
pocket-tts convert --variant english --dtype q4k --voices alba,jane
pocket-tts generate -m french-q8_0.gguf -t "Bonjour." -o out.wav
pocket-tts info -m french-q8_0.gguf                        # check a model/config

# GPU
pocket-tts generate -m french-q8_0.gguf --device cuda -t "Bonjour." -o out.wav

# Serve
pocket-tts serve -m french-q8_0.gguf --port 8000
curl -s localhost:8000/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"pocket-tts","input":"Bonjour.","voice":"estelle"}' -o out.wav
```

More: [docs/generate.md](docs/generate.md), [docs/serve.md](docs/serve.md),
[docs/rust-api.md](docs/rust-api.md), [docs/docker.md](docs/docker.md).

## Configuration

**Location**: there is no configuration file. The only state on disk is
the Hugging Face cache: `~/.cache/huggingface` on Linux and macOS,
`%USERPROFILE%\.cache\huggingface` on Windows (override with `HF_HOME`);
`.env` files are read by your shell or Docker, not by the binary. Everything is set by
command-line options (`pocket-tts <command> --help`) or environment
variables — template: [`pocket-tts.example.env`](pocket-tts.example.env),
copy to `.env` or export. To change a setting, change the option or variable
and restart the process; options take precedence over variables. Model
configs (layers, text options) are built into the binary or the GGUF file.

| Option / variable | Role | Default |
|---|---|---|
| `--model`, `POCKET_TTS_MODEL` | GGUF file | — |
| `--variant`, `POCKET_TTS_VARIANT` | Built-in model when no GGUF is given | `english` |
| `--device`, `POCKET_TTS_DEVICE` | `cpu`, `cuda`, `cuda:N`, `metal` | `cpu` |
| `--threads`, `POCKET_TTS_THREADS` | CPU threads (`RAYON_NUM_THREADS`) | `min(4, cores)` |
| `--temperature` | Sampling temperature | model's (0.3) |
| `--voice`, `POCKET_TTS_VOICE` (serve) | Default voice | language default |
| `--host`, `--port`, `POCKET_TTS_HOST/PORT` | Server address | `127.0.0.1:8000` |
| `HF_TOKEN` | Hugging Face token for gated weights | `hf auth login` cache |
| `RUST_LOG` | Log level | `info` |

**Verify** a configuration with `pocket-tts info` (same options as
`generate`): it loads the model and prints variant, device, threads, layers,
temperature and available voices.

## HTTP API

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | | `{"status":"healthy","version":…}` once the model is loaded |
| `POST /generate` | `{"text", "voice"?, "temperature"?}` | WAV |
| `POST /stream` | same | streamed PCM |
| `POST /tts` | multipart form (Python-server compatible) | WAV |
| `POST /v1/audio/speech` | OpenAI `{"model", "input", "voice"?}` | WAV |

`voice` accepts a predefined name, a server-side path, an `hf://` URL or
base64 WAV data.

## Performance and quality

Full figures and method: [docs/performance.md](docs/performance.md).
Parity with Python at temperature 0 (same length, correlation ≥ 0.9998 on
short prompts) and the list of upstream changes ported:
[docs/porting-status.md](docs/porting-status.md). ASR word error rate on 20
English and 20 French sentences is the same for f32, q8_0 and q4k.

## Tests

```bash
cargo test --release --workspace          # unit + integration (model tests need network)
tests/e2e.sh                              # release binary end to end
scripts/parity/matrix.sh                  # parity with the Python reference (setup in the script)
python3 scripts/eval/quality.py --help    # intelligibility through an ASR endpoint
```

Requirements, traceability and manual tests: [spec/](spec/specification.md).

## Known limitations

- Not bit-exact with Python: threaded float reductions differ and the
  autoregressive loop amplifies it on long chunks (same length and EOS).
- Do not pin the process to a few CPUs (`taskset`): one kernel thread pool
  spins and a frame then takes ~1 s.
- Metal compiles but is untested; only English and French were evaluated
  for quality.
- Voice cloning needs the gated Hugging Face weights.
- One request at a time per server (generation is serialized).

## Roadmap

Details and context: [TODO.md](TODO.md).

- Batch several requests on GPU (the GPU is launch-bound at batch 1).
- CUDA graphs for the decode step.
- Quality evaluation of German, Italian, Spanish, Portuguese and Dutch.
- zallama backend entry.

## Project

- [CHANGELOG.md](CHANGELOG.md) · [CREDITS.md](CREDITS.md) · [CONTRIBUTING.md](CONTRIBUTING.md)
- [spec/specification.md](spec/specification.md) · [spec/matrix.md](spec/matrix.md) · [spec/manual-tests.md](spec/manual-tests.md)
- [portfolio/](portfolio/README.md)

## License

Code: MIT OR Apache-2.0 (see [LICENSE](LICENSE)). Model weights: CC-BY-4.0
(Kyutai) — credit Kyutai when distributing converted GGUF files.
