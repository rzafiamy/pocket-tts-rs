# Docker

The `Dockerfile` builds a CPU image with the `pocket-tts` binary; models are
downloaded at runtime (or mounted as GGUF files).

```bash
docker build -t pocket-tts .

# Built-in variant, Hugging Face cache shared with the host
docker run --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e POCKET_TTS_VARIANT=french pocket-tts

# A converted GGUF, no network needed once the voice is embedded
docker run --rm -p 8000:8000 -v "$PWD/models:/models" \
  -e POCKET_TTS_MODEL=/models/french-q8_0.gguf pocket-tts

# One-off generation
docker run --rm -v "$PWD:/out" pocket-tts generate --variant english -t "Hello." -o /out/hello.wav
```

Gated voice-cloning weights need `-e HF_TOKEN=…` (or the mounted cache of a
logged-in host). Every `POCKET_TTS_*` variable from
[`pocket-tts.example.env`](../pocket-tts.example.env) works, e.g. with
`--env-file .env`.

GPU images need `--features cuda`, an `nvidia/cuda:*-devel` builder image
and `--gpus all` at run time; they are not provided here.
