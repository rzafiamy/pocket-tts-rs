#!/usr/bin/env bash
# tests/e2e.sh — end-to-end check of the release binary: GGUF conversion,
# synthesis from the GGUF (embedded voice), and the HTTP server
# (/health, /v1/audio/speech). Needs network access to Hugging Face on the
# first run (weights are cached afterwards).
#
#   tests/e2e.sh [binary]      default: target/release/pocket-tts
#
# covers: REQ-GGF-001, REQ-GGF-003, REQ-CLI-001, REQ-SRV-001
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="${1:-target/release/pocket-tts}"
[ -x "$BIN" ] || { echo "binary not found: $BIN (run ./build.sh or cargo build --release)" >&2; exit 1; }
WORK=$(mktemp -d)
PORT=${PORT:-18977}
SERVER=""
cleanup() { [ -n "$SERVER" ] && kill "$SERVER" 2>/dev/null; rm -rf "$WORK"; }
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
wav_seconds() { python3 -c "import wave,sys; w=wave.open(sys.argv[1]); print(w.getnframes()/w.getframerate())" "$1"; }

echo "== convert french q8_0"
"$BIN" convert --variant french --dtype q8_0 -o "$WORK/fr.gguf" >/dev/null || fail "convert"
size=$(stat -c %s "$WORK/fr.gguf")
[ "$size" -lt 180000000 ] || fail "q8_0 file too large: $size bytes"

echo "== generate from the GGUF (embedded voice)"
"$BIN" generate -q -m "$WORK/fr.gguf" -t "Bonjour, ceci est un test de bout en bout." -o "$WORK/a.wav" || fail "generate"
secs=$(wav_seconds "$WORK/a.wav")
python3 -c "import sys; s=float(sys.argv[1]); sys.exit(0 if 1.0 < s < 10.0 else 1)" "$secs" || fail "unexpected duration $secs s"

echo "== serve"
"$BIN" serve -m "$WORK/fr.gguf" --port "$PORT" --warmup false >"$WORK/server.log" 2>&1 &
SERVER=$!
for _ in $(seq 60); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 0.5; done
curl -sf "localhost:$PORT/health" >/dev/null || { cat "$WORK/server.log"; fail "server did not start"; }
code=$(curl -s -o "$WORK/b.wav" -w '%{http_code}' "localhost:$PORT/v1/audio/speech" \
  -H 'content-type: application/json' \
  -d '{"model":"pocket-tts","input":"Le serveur répond.","voice":"estelle"}')
[ "$code" = 200 ] || fail "/v1/audio/speech returned $code"
secs=$(wav_seconds "$WORK/b.wav")
python3 -c "import sys; s=float(sys.argv[1]); sys.exit(0 if 0.5 < s < 8.0 else 1)" "$secs" || fail "unexpected duration $secs s"

echo "e2e OK"
