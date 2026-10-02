# Traceability matrix

Requirement ([specification.md](specification.md)) → feature → module → test.
Automated tests carry the ID in a `/// covers: REQ-…` comment; manual tests
are described in [manual-tests.md](manual-tests.md).

| ID | Requirement | Feature | Module | Test | Status | Notes |
|---|---|---|---|---|---|---|
| REQ-CFG-001 | Built-in configs | Variants | `crates/pocket-tts/src/builtin_configs.rs`, `tts_model.rs` | `builtin_configs_parse` | ✅ | 20 configs |
| REQ-TXT-001 | Upstream text preparation | Text | `crates/pocket-tts/src/text_chunking.rs` | `terminal_punctuation`, `prepare_basic`, `prepare_padding_and_semicolons`, `prepare_replace_characters`, manual MT-01 | ✅ | chunk ids equal upstream |
| REQ-TXT-003 | Text normalization (fr/en numbers, Markdown) | Text | `tn` crate ([rzafiamy/tn-rs](https://github.com/rzafiamy/tn-rs), its own tests and matrix), `crates/pocket-tts/src/normalize.rs`, `tts_model.rs` (`split_into_best_sentences`) | `variant_languages`, `model_languages_are_normalized` | ✅ | ASR on a 2.4k-char chat text: digits read correctly |
| REQ-TXT-002 | Explicit pauses | Pauses | `crates/pocket-tts/src/pause.rs`, `tts_model.rs` | `test_parse_explicit_pause_ms`, `test_strip_pause_markers`, `test_generate_with_pauses_adds_silence` | ✅ | |
| REQ-VOI-001 | Predefined voices | Voices | `crates/pocket-tts/src/voices.rs`, `tts_model.rs` | `import_exported_model_state`, manual MT-01 | ✅ | |
| REQ-VOI-002 | Voice cloning | Voices | `crates/pocket-tts/src/tts_model.rs`, `audio.rs` | `test_voice_cloning_from_ref_wav`, manual MT-01 | ✅ | needs gated weights |
| REQ-INF-001 | Parity with Python | Inference | `crates/pocket-tts/src/tts_model.rs`, `models/` | manual MT-01 | ✅ | `scripts/parity/matrix.sh` |
| REQ-INF-002 | Streaming and EOS | Inference | `crates/pocket-tts/src/tts_model.rs` | `test_audio_generation_produces_valid_output`, manual MT-01 | ✅ | |
| REQ-GGF-001 | Self-contained GGUF | GGUF | `crates/pocket-tts/src/gguf.rs`, `crates/pocket-tts-cli/src/commands/convert.rs` | `f32_gguf_matches_safetensors`, `tests/e2e.sh` | ✅ | |
| REQ-GGF-002 | f32 GGUF = safetensors | GGUF | `crates/pocket-tts/src/gguf.rs` | `f32_gguf_matches_safetensors` | ✅ | |
| REQ-GGF-003 | Quantized GGUF | GGUF | `crates/pocket-tts/src/gguf.rs`, `modules/linear.rs` | `q8_gguf_is_small_and_speaks`, `tests/e2e.sh` | ✅ | 155 MB q8_0 |
| REQ-QUA-001 | No intelligibility loss | Quantization | `crates/pocket-tts/src/modules/linear.rs` | manual MT-02 | ✅ | WER table in docs/performance.md |
| REQ-OPS-001 | Optimized kernels exact | Kernels | `crates/pocket-tts/src/modules/conv.rs`, `activations.rs`, `sdpa.rs` | `im2col_conv_matches_candle`, `matmul_convtr_matches_reference`, `depthwise_convtr_matches_grouped_kernel`, `matches_candle_gelu`, `test_sdpa_handles_non_contiguous_inputs`, `test_generate_mask_chunk_window` | ✅ | |
| REQ-PRF-001 | CPU speed | Performance | `crates/pocket-tts/src/modules/` | manual MT-03 | ✅ | ~7× |
| REQ-GPU-001 | CUDA speed and VRAM | GPU | `crates/pocket-tts/src/modules/`, `crates/pocket-tts-cli/src/loader.rs` | manual MT-04 | ✅ | 28×, 680 MiB |
| REQ-SRV-001 | HTTP API | Server | `crates/pocket-tts-cli/src/server/` | `test_api_full_flow`, `tests/e2e.sh` | ✅ | |
| REQ-CLI-001 | CLI generate | CLI | `crates/pocket-tts-cli/src/commands/generate.rs` | `test_cli_generate_basic`, `tests/e2e.sh` | ✅ | |

Updated: 2026-10-02 (v0.7.0).
