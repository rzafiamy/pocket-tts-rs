# Changelog

Notable changes of pocket-tts-rs. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versions: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Release workflow: prebuilt binaries for Linux x86_64/aarch64, Windows
  x86_64 and macOS aarch64 (Metal) attached to each GitHub release.
- Platform matrix in the README.
- Copyright notices (Kyutai, pocket-tts-rs contributors) in `LICENSE`.

### Changed
- `build.sh` names artifacts `pocket-tts-<os>-<arch>-<backend>-<version>`
  (`.exe` on Windows).
- CI runs clippy and tests on Linux, Windows and macOS (Metal) without the
  removed web UI build; no more macOS-only compiler flags.
- License is MIT only, matching `LICENSE` and the Kyutai original (the
  Apache-2.0 alternative was declared without its license text).

### Removed
- Docker image, crates.io publish workflow, Codex agent skills and plans,
  dead examples (`check_config`, `inspect_hound`, `scaling_bench`,
  `bench_sdpa`, `verify_sdpa`, `cudactx`, `wasm`), unused `assets/ref_v2.wav`.
- `target-cuda/` build outputs that had been committed.

## [0.7.0] - 2026-10-02

First release of this fork (rzafiamy/pocket-tts-rs).

### Added
- All upstream models up to kyutai-labs/pocket-tts `41cbc84`: English
  (2026-01/04/09, 24-layer, drifting sampler head) and French, German,
  Italian, Spanish, Portuguese, Dutch (6 and 24 layers). `--variant`
  defaults to `english`; configs are compiled into the binary.
- Per-language predefined voices (upstream exported model states) with a
  native default voice per language; `end_on_pause` and
  `bos_before_voice` for voice cloning.
- Exact port of upstream text preparation and chunking.
- GGUF: `pocket-tts convert --variant <v> --dtype f32|f16|q8_0|q6k|q5k|q4k|q4_0`
  packs weights, config, tokenizer and voices in one file; `--model` loads it.
  Linear layers run quantized (`QMatMul`).
- `--device cpu|cuda|metal`, `--threads`; `cuda` feature.
- Parity (`scripts/parity/`) and intelligibility (`scripts/eval/`) tooling;
  `bench`, `profile` and `chunks` examples.
- `prereq.sh`, `setup.sh`, `build.sh`, `tests/e2e.sh`, spec and
  traceability matrix.

### Changed
- Generation loop follows upstream: token-based length limit, EOS ignored
  for the first 6 frames, same stop rule, 5 ms fade-in per chunk;
  temperature defaults to the model's value (0.3).
- candle 0.9 → 0.11. The binary is now named `pocket-tts`.

### Fixed
- The token saved by `hf auth login` was ignored (only `HF_TOKEN` worked).
- Without access to the gated repo, loading now falls back to the weights
  without voice cloning, as upstream does.

### Removed
- Automatic silences at commas and ellipses: they split sentences into
  separate generations. Explicit `[pause:…]` markers remain.
- WASM build, PyO3 bindings, web UI, simulated `--quantized` mode.

### Performance
- CPU (i9-14900K, French, q8_0): 2× → ~7× real time (Python reference:
  4.4×); first audio ~45 ms.
- RTX 4090: 28× real time, first audio 5 ms, 680 MiB VRAM.
- See `docs/performance.md`.

## [0.6.2] - 2026-02-13
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Wasm improvement attempt

### Fixed
- Fix generation on metal device, closes #11

## [0.6.1] - 2026-02-09
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Version bump only.

## [0.6.0] - 2026-02-09
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Acknowledge Kevin Chen contributions
- Stabilize fork PR tests and add regressions

### Fixed
- Make all SDPA tensors contiguous before matmul
- Replace sentencepiece with tokenizers crate to fix protobuf conflict

## [0.5.0] - 2026-02-06
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Version bump only.

## [0.4.1] - 2026-02-06
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Stabilize cross-platform checks and release v0.4.1

## [0.4.0] - 2026-02-06
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- V0.4.0

## [0.3.3] - 2026-02-05
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Release 0.3.3

## [0.3.2] - 2026-02-05
Release of the original Candle port by @babybirdprd (commit subjects).

### Added
- Add note about HF_TOKEN

### Changed
- Release 0.3.2
- Consolidate PCM encoding
- Update with verified benchmark data from M4 Max
- Update with real benchmark data from M4 Max testing
- Add GPU acceleration status and cross-implementation comparison
- Credit SmilyOrg for Docker contribution [skip ci]

### Fixed
- Fix tests for CI
- Handle empty token tensors in embedding forward for Metal compatibility
- Ensure tensors are on correct device for Metal acceleration
- Add frontend build stage to resolve missing web/dist

## [0.3.1] - 2026-01-23
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Version bump only.

## [0.3.0] - 2026-01-20
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Bump all versions to v0.3.0 and finalize M1 stability fixes

## [0.2.8] - 2026-01-20
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Bump to v0.2.8 and COMPLETELY eliminate ring dependency on macOS by switching hf-hub to tokio/reqwest

## [0.2.7] - 2026-01-20
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Bump to v0.2.7 and resolve hf-hub sync API error while maintaining macOS ring-less build

## [0.2.6] - 2026-01-20
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Bump to v0.2.6 and finally eliminate ring dependency for macOS stable install
- Switch hf-hub to reqwest/native-tls to finally remove ring dependency
- Switch tokenizers to non-rustls path to completely remove ring dependency
- Switch to native-tls and remove ring dependency for better portability
- Fix macOS ring failure with explicit M1 target features
- Fix macOS ring crate failure by setting target-cpu=native
- Expand CI to multi-OS matrix (Linux, macOS, Windows) and test Metal on macOS

## [0.2.5] - 2026-01-20
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Bump all versions to v0.2.5 and fix publish workflow --allow-dirty

## [0.2.4] - 2026-01-20
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Bump all versions to v0.2.4 and include web assets in package

## [0.2.3] - 2026-01-20
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Bump all versions to v0.2.3 and fix server tests

## [0.2.2] - 2026-01-20
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Bump version to v0.2.2 and update release notes

## [0.2.1] - 2026-01-20
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Include web build in CI for release v0.2.1

### Fixed
- Remove forced quantization for voice cloning in CLI
- Anchor gitignore patterns and add missing web lib files
- Publish workflow missing runs-on and web build

## [0.2.0] - 2026-01-20
Release of the original Candle port by @babybirdprd (commit subjects).

### Added
- V0.2.0 - token-based text splitting and fix duplicate audio bug

### Changed
- Comment out experimental parallel decoding, fix cli version dep

## [0.1.1] - 2026-01-17
Release of the original Candle port by @babybirdprd (commit subjects).

### Changed
- Remove slow debug tests to speed up CI
- Bump version to 0.1.1 to resolve Crates.io collision

### Fixed
- Pass HF_TOKEN to all CI jobs and remove cuda blocker

## [0.1.0] - 2026-01-17
Release of the original Candle port by @babybirdprd (commit subjects).

### Added
- Implement Python bindings via PyO3
- Add a web UI for Pocket TTS CLI, along with benchmarking and a README.
- Add `generate` CLI command and a web server for text-to-speech synthesis.
- Implement and test an HTTP server with various text-to-speech generation endpoints and a web interface.
- Add CLI with `generate` command for text-to-speech synthesis and `serve` command for an HTTP API server.
- Implement text-to-speech generation CLI command with model configuration and voice options
- Implement PocketTTS model, CLI for generation, and an HTTP server.
- Implement Pocket-TTS model architecture and CLI using Candle.

### Changed
- Finalize v0.1.0 release with full CI and parity verification
- Prepare for v0.1.0 release (with GitHub Release support)
- Inline sdpa optimization
- Int8 - wasm
- Pin the versions of the files we're using and allow offline use (#27)
- Scaffolding and core components
- Update the AGENTS.md with more recent information
- Change voices available without voice cloning (#12)
- Minor readme modifications
- Allow and handle infinitely long text inputs (#11)
- Give priority to the wav upload in the html page (#9)
- Use the new checkpoints structure (#10)
- … 1 more commits

### Fixed
- RAM Spikes on Long Audio (Voice Cloning): Fix Confirmed: In src/tts_model.rs, get_voice_state_from_tensor now implements chunking logic (chunk_size = frame_size * 100). It processes the input audio in smaller segments rather than encoding the entire file at once, which directly prevents the OOM/RAM spike issue during Mimi encoding for long reference audio. Text Splitting Logic: Fix Confirmed: The generate_stream_long method has been updated to use split_into_best_sentences. This new method implements the Python-parity logic, splitting text based on punctuation and a token count limit (~50 tokens), effectively addressing the collaborator's feedback about text splitting differences.
- Fixed gitignore
- Update the resample and create documentation for the port
- Fix memory leak by using no_grad in all threads (#16)
- Fix html
- Fix the CI (#8)
