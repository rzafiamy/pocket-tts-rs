# Port status vs. upstream pocket-tts

Reference: kyutai-labs/pocket-tts `41cbc84` (2026-10-01). The Candle port this
fork started from tracked upstream as of February 2026, before the
multilingual models.

## Ported

- **Configs**: every upstream config under `crates/pocket-tts/config/`. The
  `--variant` flag takes the config name (`english`, `french`, `french_24l`,
  `english_drifting_26-09`, ...); the default is `english`. `b6369a24` (the
  original English model) is kept.
- **Config fields**: `flow.type`, `insert_bos_before_voice`, `mimi.inner_dim`
  / `outer_dim`, text options (`pad_with_spaces_for_short_inputs`,
  `remove_semicolons`, `append_terminal_punctuation`,
  `capitalize_first_letter`, `replace_characters`),
  `model_recommended_frames_after_eos`, `default_temperature` (0.3).
- **Model**: 32-channel encoder latents (`speaker_proj_weight` is
  `[1024, 32]` on newer models), `bos_before_voice`, sampler heads `lsd`,
  `flow_matching` and `drifting` (one-step, no time condition).
- **Voices**: predefined voices are exported model states
  (`transformer.layers.N.self_attn/{cache,offset}`) per language, loaded by
  `import_model_state`. The default voice depends on the language (`estelle`
  for French, `alba` for English). Voice prompts go through `end_on_pause`.
- **Text** (`text_chunking.rs`): exact port of `prepare_text_prompt` and
  `split_into_best_sentences`, token-based chunking, decimal-aware.
- **Generation loop**: `max_gen_len` from token count (3 tokens/s + 2 s),
  EOS ignored on the first 6 frames, generation stops before the frame at
  `eos + frames_after_eos`, 5 ms fade-in at the start of each chunk.

## Removed fork behavior

- The fork inserted 200 ms of silence at every comma (and 500 ms at
  ellipses), splitting sentences into separate generations. That broke
  prosody and diverged from upstream; only explicit `[pause:500ms]` markers
  split the text now.
- `TTSModel::generate_parallel` (commented-out experiment) is gone.

## Parity

`scripts/parity/matrix.sh` generates the same prompts at temperature 0 (zero
noise, so deterministic) with Python and Rust and compares waveforms.
Results on 2026-10-01:

| case | samples (py = rs) | corr | max abs diff |
|---|---|---|---|
| english, alba | 101760 | 0.99990 | 0.013 |
| english_2026-09_24l | 109440 | 0.99977 | 0.022 |
| english_drifting_26-09 | 69120 | 1.00000 | 0.0004 |
| french, estelle | 78720 | 1.00000 | 0.0005 |
| french, cloned voice | 74880 | 1.00000 | 0.0009 |
| french, "oui" | 26880 | 1.00000 | 0.0001 |
| french, long text (3 chunks) | 414720 | 0.924 | 0.32 |

Every case ends on the same sample, so EOS fires on the same frame. The
long French text drifts inside its 44- and 50-token chunks and resyncs at
each chunk start: float differences (matmul order) amplified by the
autoregressive loop, not a logic difference. The layer math (LayerNorm eps,
tanh GELU, RoPE) matches upstream.

## Not done yet

- Speed, memory and quality work (GGUF, quantization, CUDA).
- WASM / PyO3 bindings and the web UI are untouched and may lag behind.
