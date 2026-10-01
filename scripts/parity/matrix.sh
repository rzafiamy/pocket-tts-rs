#!/usr/bin/env bash
# Parity matrix: generates the same prompts at temperature 0 with the Python
# reference and this port, and compares the waveforms.
#
# Setup (once):
#   git clone https://github.com/kyutai-labs/pocket-tts /tmp/pocket-tts-py
#   python3 -m venv .venv-parity
#   .venv-parity/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
#   .venv-parity/bin/pip install -e /tmp/pocket-tts-py
#   cargo build --release -p pocket-tts-cli
#
# Usage: scripts/parity/matrix.sh [out_dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PARITY_PYTHON:-$ROOT/.venv-parity/bin/python}"
BIN="$ROOT/target/release/pocket-tts"
OUT="${1:-$(mktemp -d)}"
HERE="$ROOT/scripts/parity"
mkdir -p "$OUT" || { echo "cannot create output directory $OUT" >&2; exit 1; }
[ -x "$PY" ] || { echo "Python reference not found at $PY (see setup above, or set PARITY_PYTHON)" >&2; exit 1; }
[ -x "$BIN" ] || { echo "binary not found: $BIN (cargo build --release -p pocket-tts-cli)" >&2; exit 1; }
"$PY" -c "import pocket_tts" 2>/dev/null || { echo "pocket_tts is not installed in $PY" >&2; exit 1; }

run() { # name variant voice text
  local n=$1 v=$2 vo=$3 t=$4
  "$PY" "$HERE/py_gen.py" "$v" "$vo" "$t" "$OUT/py_$n.wav" 0.0 >/dev/null 2>&1 || { echo "$n: python failed"; return; }
  "$BIN" generate --variant "$v" -v "$vo" --temperature 0 -t "$t" -o "$OUT/rs_$n.wav" >/dev/null 2>&1 || { echo "$n: rust failed"; return; }
  echo "$n: $("$PY" "$HERE/cmp.py" "$OUT/py_$n.wav" "$OUT/rs_$n.wav" | tr '\n' ' ')"
}

LONG_FR="Le prix est de 3.14 euros, ce qui est raisonnable. « Vraiment ? », demanda-t-elle ; il répondit : oui! Ensuite, nous sommes partis vers la gare, puis nous avons pris le train de nuit qui traversait les montagnes enneigées, les vallées profondes et les petits villages endormis, jusqu'à l'aube. Enfin."

run en english alba "Hello world. This is a test of the English model, version two."
run en24 english_2026-09_24l alba "Hello world. This is a test of the twenty four layer model."
run drift english_drifting_26-09 alba "Hello world. This is the drifting sampler head."
run fr french estelle "Bonjour le monde. Je suis le TTS de poche de Kyutai."
run fr_long french estelle "$LONG_FR"
run fr_short french estelle "oui"
echo "outputs in $OUT"
