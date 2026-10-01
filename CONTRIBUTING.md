# Contributing

## Setup

```bash
./setup.sh            # prerequisites, dependencies, cargo check (--cuda for GPU)
./build.sh            # release binary in build/
```

## Repository

- `crates/pocket-tts`: library (models, kernels, GGUF, text, voices).
- `crates/pocket-tts-cli`: `pocket-tts` binary (generate, serve, convert, info).
- `crates/pocket-tts/config/`: upstream configs, compiled into the binary.
- `scripts/parity/`: comparison against the Python reference.
- `scripts/eval/`: intelligibility (ASR word error rate).
- `spec/`: requirements, traceability matrix, manual tests.
- `docs/`: user docs, porting status, performance.

## Before a pull request

```bash
cargo fmt --all
cargo clippy --release --workspace --all-targets -- -D warnings
cargo test --release --workspace
tests/e2e.sh
```

- A change to the model or text handling must keep the parity matrix
  (`scripts/parity/matrix.sh`) at the values of `docs/porting-status.md`.
- A speed optimization comes with a test against a reference
  implementation (see `crates/pocket-tts/src/modules/conv.rs` tests) and a
  measurement (`cargo run --release --example profile`).
- New requirements get an ID in `spec/specification.md`, a row in
  `spec/matrix.md` and a `/// covers: REQ-…` comment on their test.
- Note user-visible changes in `CHANGELOG.md`.

## Following upstream

Diff kyutai-labs/pocket-tts against the commit named in
`docs/porting-status.md`, port behavior changes, copy new configs into
`crates/pocket-tts/config/` and add them to `src/builtin_configs.rs`, then
rerun the parity matrix.

## Coding agents

See [AGENTS.md](AGENTS.md).
