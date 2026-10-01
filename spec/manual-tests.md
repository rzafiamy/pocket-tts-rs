# Manual tests

Results of 2026-10-02 (v0.7.0), i9-14900K, RTX 4090, Ubuntu 24.04, CUDA 12.8.
Reference: kyutai-labs/pocket-tts `41cbc84`, torch 2.14 CPU.

## MT-01 — Parity with the Python reference (REQ-INF-001, REQ-TXT-001, REQ-VOI-001, REQ-VOI-002, REQ-INF-002)

1. Install the reference (see the header of `scripts/parity/matrix.sh`).
2. `cargo build --release -p pocket-tts-cli`
3. `PARITY_PYTHON=.venv-parity/bin/python scripts/parity/matrix.sh`
4. Chunking: `cargo run --release --example chunks -- french "<long text>"`
   against `scripts/parity/py_chunks.py french "<long text>"`.

Expected: same sample count for every case, correlation ≥ 0.999 on short
prompts; identical chunks and token ids. Result: ✅ (table in
`docs/porting-status.md`; long 3-chunk French text: same length, corr 0.92,
float drift); chunks and ids identical. Voice cloning from a WAV: same
length, corr 1.00000.

## MT-02 — Intelligibility of the quantized models (REQ-QUA-001)

```bash
for d in q8_0 q4k; do target/release/pocket-tts convert --variant french --dtype $d -o models/french-$d.gguf; done
python3 scripts/eval/quality.py --lang french --repeats 2 \
  --system "f32=target/release/pocket-tts generate -q --variant french -t {text} -o {out}" \
  --system "q8_0=target/release/pocket-tts generate -q -m models/french-q8_0.gguf -t {text} -o {out}" \
  --system "q4k=target/release/pocket-tts generate -q -m models/french-q4k.gguf -t {text} -o {out}"
```

ASR: Parakeet TDT 0.6B v3 through zallama (`--asr`, `--asr-model`).
Expected: quantized WER within noise of f32. Result: ✅ FR 4.7 / 4.7 / 4.2 %,
EN 3.2 / 3.2 / 2.8 % (f32 / q8_0 / q4k), Python 2.5 / 3.2 %.

## MT-03 — CPU speed (REQ-PRF-001)

`cargo run --release --example bench -- models/french-q8_0.gguf "" 3`

Expected: ≥ 5× real time, first chunk < 100 ms. Result: ✅ ~7× (6.99, 7.19),
first chunk ~45 ms.

## MT-04 — GPU speed and VRAM (REQ-GPU-001)

```bash
./build.sh --cuda
cargo run --release --features cuda --example bench --target-dir target-cuda -- models/french-q8_0.gguf "" 6 cuda
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # while it runs
target-cuda/release/pocket-tts generate -m models/french-f32.gguf --device cuda --temperature 0 -t "..." -o gpu.wav
```

Expected: ≥ 20× real time, < 1 GiB, output equal to the Python reference
at temperature 0. Result: ✅ 28×, first chunk 5 ms, 680 MiB peak (q8_0),
corr 1.00000 with Python.

## MT-05 — zallama backend contract (REQ-SRV-001)

```bash
target/release/pocket-tts serve -m models/french-q8_0.gguf --port 18977 &
until curl -sf localhost:18977/health; do sleep 1; done
curl -s localhost:18977/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"pocket-tts","input":"Bonjour.","voice":"estelle"}' -o out.wav
```

Expected: `/health` answers once the model is loaded; a WAV comes back.
Result: ✅ (also automated in `tests/e2e.sh`).
