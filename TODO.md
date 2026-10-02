# TODO

Next steps after v0.7.0, in priority order. Measurements behind them:
[docs/performance.md](docs/performance.md).

## GPU

- [ ] **Batch concurrent requests on GPU.** At batch 1 a frame costs ~2.7 ms
  (FlowLM 1.4 + Mimi 1.3) and is bound by kernel launches, not compute: the
  GPU can generate several streams for nearly the same cost per frame. Needs
  per-request KV caches stacked on the batch dimension, a scheduler in the
  server (it serializes requests today), and EOS handled per stream.
  Watch out: candle 0.11's `conv_transpose1d` is wrong for batch > 1 (our
  matmul path is not) and `matmul` with a stride-0 batch is wrong.
- [ ] **CUDA graphs for the decode step.** Capture the per-frame FlowLM +
  sampler + Mimi kernels once and replay them, to cut launch overhead
  (target: < 1 ms per frame at batch 1). malaga does this for NLLB.

## Integration

- [x] **zallama backend entry**: `pocket-tts-server` in zallama v1.24.0
  (`build-pocket-tts.sh`, models `pocket-tts-fr` / `pocket-tts-en`, 808 MiB
  VRAM with the 27 voices embedded).
- [ ] Unknown `voice` returns HTTP 500; it should be a 400 (client error).
- [ ] Publish GGUF files (english/french q8_0, q4k) with CC-BY-4.0 credit
  to Kyutai.

## Quality and coverage

- [ ] **Stronger text normalization.** `normalize.rs` covers the common
  French/English cases with regexes. Next: a grammar-based normalizer in the
  spirit of NVIDIA NeMo text processing (WFST classify/verbalize: dates,
  measures, money, addresses, roman numerals, context-dependent readings),
  ported to Rust, plus a pronunciation dictionary for recent words, names
  and brands the model has not seen (word → respelling, user-editable).
  German, Italian, Spanish, Portuguese and Dutch rules.
- [ ] Quality evaluation (WER) of German, Italian, Spanish, Portuguese,
  Dutch; parity matrix for the 24-layer French model.
- [ ] Measure the Metal build (speed, parity) and CUDA on Windows.
- [ ] Report the candle 0.11 bugs upstream (grouped `conv_transpose1d`
  speed and batch > 1 results; stride-0 batch `matmul`).
