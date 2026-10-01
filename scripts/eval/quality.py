#!/usr/bin/env python3
"""Intelligibility check: synthesize sentences, transcribe them with an
OpenAI-compatible ASR endpoint, report word error rate (WER) per system.

Systems are given as NAME=COMMAND templates, run once per sentence with
{text} and {out} substituted, e.g.
  rs_q8="target/release/pocket-tts-cli generate -q -m models/french-q8_0.gguf -t {text} -o {out}"

Usage: quality.py --lang french --asr http://localhost:6767 --asr-model parakeet-tdt-v3-cpu \
         --system NAME=CMD [--system ...] [--repeats 2]
"""
import argparse, json, os, re, shlex, subprocess, sys, tempfile, unicodedata, urllib.request, uuid

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("-", " ")
    text = re.sub(r"[^\w' ]+", " ", text)
    return text.split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)], len(r)


def transcribe(url, model, wav):
    boundary = uuid.uuid4().hex
    with open(wav, "rb") as f:
        audio = f.read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--asr", default="http://localhost:6767")
    ap.add_argument("--asr-model", default="parakeet-tdt-v3-cpu")
    ap.add_argument("--system", action="append", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--keep", help="directory to keep the wavs in")
    args = ap.parse_args()

    sentences = json.load(open(os.path.join(HERE, "sentences.json")))[args.lang]
    workdir = args.keep or tempfile.mkdtemp()
    os.makedirs(workdir, exist_ok=True)
    for spec in args.system:
        name, cmd = spec.split("=", 1)
        errors = words = failed = 0
        for rep in range(args.repeats):
            for i, text in enumerate(sentences):
                out = os.path.join(workdir, f"{name}_{rep}_{i}.wav")
                full = cmd.replace("{text}", shlex.quote(text)).replace("{out}", shlex.quote(out))
                if subprocess.run(full, shell=True, capture_output=True).returncode != 0 or not os.path.exists(out):
                    failed += 1
                    continue
                hyp = transcribe(args.asr, args.asr_model, out)
                e, n = wer(text, hyp)
                errors += e
                words += n
                if e:
                    print(f"  [{name}] {e}/{n}: {hyp}", file=sys.stderr)
        rate = 100.0 * errors / max(words, 1)
        print(f"{name}: WER {rate:.1f}% ({errors}/{words} words, {failed} failed)")


if __name__ == "__main__":
    main()
