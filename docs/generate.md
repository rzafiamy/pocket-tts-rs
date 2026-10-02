# `pocket-tts generate`

Synthesizes text to a WAV file (or raw PCM on stdout with `--stream`).

```bash
pocket-tts generate --variant french -t "Bonjour le monde." -o out.wav
pocket-tts generate -m french-q8_0.gguf -t "Bonjour." -o out.wav
pocket-tts generate --variant english --stream -t "Hello." | ffplay -f s16le -ar 24000 -ac 1 -
```

Without `--text`, a greeting in the model's language is spoken.

## Options

| Option | Meaning | Default |
|---|---|---|
| `-t, --text` | Text; `[pause:500ms]` / `[pause:1s]` insert silence | greeting |
| `-v, --voice` | Voice (see below) | language default |
| `-o, --output` | WAV path | `output.wav` |
| `--variant` | Built-in model (`english`, `french`, `french_24l`, `english_drifting_26-09`, …) or config YAML | `english` |
| `-m, --model` | GGUF file (overrides `--variant`) | — |
| `--device` | `cpu`, `cuda`, `cuda:N`, `metal` | `cpu` |
| `--threads` | CPU threads | `min(4, cores)` |
| `--temperature` | Sampling temperature | model's (0.3) |
| `--lsd-decode-steps` | Sampler steps (more is slower; 1 is what Kyutai ships) | `1` |
| `--eos-threshold` | End-of-speech threshold (lower = longer) | `-4.0` |
| `--noise-clamp` | Clamp sampling noise to ±x | off |
| `--no-normalize`, `POCKET_TTS_NO_NORMALIZE` | Keep digits, symbols and Markdown as written (see `normalize.rs`) | normalization on |
| `--tight-pauses true\|false`, `POCKET_TTS_TIGHT_PAUSES` | Shorten the silence between generated chunks (320 ms after a sentence, 160 ms after a comma split) | `true` |
| `--frames-after-eos` | Frames kept after end of speech | model or length-based guess |
| `--stream` | Raw 16-bit PCM to stdout | off |
| `-q, --quiet` | Errors only | off |

Every option also has an environment variable for the model selection
(`POCKET_TTS_MODEL`, `POCKET_TTS_VARIANT`, `POCKET_TTS_DEVICE`,
`POCKET_TTS_THREADS`).

## Voices

- **Predefined name**: 27 voices per language (`alba`, `estelle`, `jean`,
  `marius`, `lola`, …; see `pocket_tts::voices::PREDEFINED_VOICES`). Each
  language has a native default: `estelle` (French), `juergen` (German),
  `giovanni` (Italian), `lola` (Spanish), `rafael` (Portuguese), `daan`
  (Dutch), `alba` (English). Voices embedded in a GGUF are used without
  download.
- **WAV file**: clones the voice (needs the gated weights). The prompt is
  resampled to 24 kHz, trimmed to end on a short pause, and encoded.
- **`.safetensors`**: an exported voice state (upstream `export-voice`
  format) or a latent prompt (`audio_prompt`).
- **`hf://owner/repo/file[@rev]`**: downloaded, then handled as above.

## Text

Text is prepared like upstream: characters absent from the training data
are replaced (French quotes, curly apostrophes…), the first letter is
capitalized, terminal punctuation is added, and long text is split into
chunks of at most 50 tokens on sentence boundaries. Each chunk is generated
from the voice state; numbers and abbreviations are read by the model as is.
