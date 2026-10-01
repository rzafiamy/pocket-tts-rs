# `pocket-tts serve`

HTTP server with the model loaded once. Requests are processed one at a time.

```bash
pocket-tts serve -m french-q8_0.gguf --host 0.0.0.0 --port 8000
pocket-tts serve --variant english --voice jane --device cuda
```

## Options

Model selection is the same as `generate` (`--variant`, `-m/--model`,
`--device`, `--threads`, `--temperature`, `--lsd-decode-steps`,
`--eos-threshold`, `--noise-clamp`), plus:

| Option | Variable | Meaning | Default |
|---|---|---|---|
| `--host` | `POCKET_TTS_HOST` | Bind address | `127.0.0.1` |
| `-p, --port` | `POCKET_TTS_PORT` | Port | `8000` |
| `--voice` | `POCKET_TTS_VOICE` | Default voice | language default |
| `--voice-cache-capacity` | | Resolved voices kept in memory | `64` |
| `--prewarm-voices` | | Extra voices loaded at startup (`alba,jean`) | none |
| `--warmup` | | Generate once at startup | `true` |

## Endpoints

### `GET /health`

`{"status":"healthy","version":"0.7.0"}`, answered once the model and the
default voice are loaded.

### `POST /generate`

```bash
curl -s localhost:8000/generate -H 'content-type: application/json' \
  -d '{"text":"Bonjour.","voice":"estelle","temperature":0.3}' -o out.wav
```

Fields: `text` (required), `voice`, `temperature`, `lsd_steps`,
`eos_threshold`, `noise_clamp`. Returns a 24 kHz mono WAV.

### `POST /stream`

Same body; returns raw 16-bit PCM as it is generated (first bytes after
~40 ms on CPU, 5 ms on GPU).

### `POST /tts`

Multipart form compatible with the Python server: `text`, and either
`voice_url` or a `voice_wav` file.

### `POST /v1/audio/speech`

OpenAI-compatible:

```bash
curl -s localhost:8000/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"pocket-tts","input":"Bonjour.","voice":"estelle"}' -o out.wav
```

`model` is accepted and ignored (one model per server); the response is WAV.

## Voices in requests

`voice` takes a predefined name, a path on the server, an `hf://` URL, or
base64 WAV (`data:audio/wav;base64,…`) for cloning. Resolved voices are
cached (LRU) by name, path + modification time, or content hash.

## Errors

Errors come back as `{"error": "…"}`.

| Status | Cause |
|---|---|
| 400 | `/tts` without text |
| 500 | Unknown or unreadable voice, generation failure (message in the body) |
