# Copyright (c) 2025-2026 Sunet.
# Contributor: Kristofer Hallin
#
# This file is part of Sunet Scribe.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Transcribe every clip of a manifest on its own and compare with its reference.

    uv run python scripts/fetch_fleurs.py DIR --count 50
    uv run python scripts/benchmark_clips.py DIR/clean.tsv --language swedish

The manifest has one "audio path<TAB>reference text" line per clip, as
fetch_fleurs.py writes it. For each variant it prints the word error rate of
every clip as it goes, the total over all clips (errors divided by all
reference words, so long clips weigh more), the speed, and the worst clips
with what differs. For regular measuring use eval.py, which is faster (clips
are joined so that Whisper's 30 second windows are full) and keeps a baseline;
this one is for looking at single clips.

Variants are those of evallib.VARIANTS: old (no fallback or filter), new (as
in production), novad, beam5. The model defaults to the one the worker uses
for the language.
"""

import argparse
import json
import time
from pathlib import Path

from evallib import VARIANTS, Decoder, default_model, describe, align, load_audio, log, normalise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("manifest")
    parser.add_argument("--model", default="", help="default: the worker's model for the language")
    parser.add_argument("--language", default="swedish")
    parser.add_argument("--variants", default="new", help="comma separated; each one is a full pass")
    parser.add_argument("--limit", type=int, default=0, help="only the first N clips")
    parser.add_argument("--worst", type=int, default=5, help="worst clips to print")
    parser.add_argument("--json", default="", help="also write the results here")
    args = parser.parse_args()

    lines = [l for l in Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    lines = lines[: args.limit] if args.limit else lines

    model = args.model or default_model(args.language)
    log(f"loading {model} ...")
    decoder = Decoder(model, args.language)
    log(f"model loaded on {decoder.device}; filter {'on' if decoder.filter_enabled else 'off'}, "
        f"temperatures {decoder.temperatures}")

    clips = []

    for line in lines:
        path, text = line.split("\t", 1)
        clips.append((path, normalise(text, args.language), load_audio(path)))

    seconds_total = sum(len(a) for _, _, a in clips) / 16000
    windows = sum(-(-len(a) // (30 * 16000)) for _, _, a in clips)
    log(f"{len(clips)} clips, {seconds_total:.0f} s of audio = {windows} Whisper windows of 30 s")

    results = {}

    for name in [v for v in args.variants.split(",") if v in VARIANTS]:
        overrides, use_filter = VARIANTS[name]
        rows, elapsed = [], 0.0
        log(f"variant '{name}': {len(clips)} clips")

        for number, (path, reference, audio) in enumerate(clips, 1):
            started = time.monotonic()
            words, _ = decoder.transcribe(audio, overrides, use_filter)
            took = time.monotonic() - started
            elapsed += took

            hypothesis = [w["t"] for w in words]
            s, d, n = align(reference, hypothesis)
            rows.append({
                "clip": Path(path).name, "words": len(reference), "sub": s, "del": d, "ins": n,
                "wer": 100 * (s + d + n) / max(len(reference), 1),
                "diff": describe(reference, hypothesis),
            })

            done = sum(len(a) for _, _, a in clips[:number]) / 16000
            errors = sum(r["sub"] + r["del"] + r["ins"] for r in rows)
            ref_words = sum(r["words"] for r in rows)
            log(f"  [{number}/{len(clips)}] {Path(path).name}  {len(audio) / 16000:.1f}s audio in {took:.1f}s  "
                f"WER {rows[-1]['wer']:.1f}%  running {100 * errors / max(ref_words, 1):.1f}%  "
                f"eta {elapsed / done * (seconds_total - done):.0f}s")

        results[name] = {"rows": rows, "seconds": elapsed}

    for name, data in results.items():
        rows = data["rows"]
        errors = sum(r["sub"] + r["del"] + r["ins"] for r in rows)
        words = sum(r["words"] for r in rows)
        print(f"\n== {name}: WER {100 * errors / max(words, 1):.2f}%  "
              f"(sub {sum(r['sub'] for r in rows)}, del {sum(r['del'] for r in rows)}, "
              f"ins {sum(r['ins'] for r in rows)} of {words} words)  "
              f"{seconds_total / max(data['seconds'], 1e-9):.1f}x realtime")

        for row in sorted(rows, key=lambda r: -r["wer"])[: args.worst]:
            print(f"  {row['wer']:5.1f}%  {row['clip']}  {row['diff'][:200]}")

    if len(results) > 1:
        print("\nper clip WER %: " + "  ".join(f"{name:>7}" for name in results))

        for index, (path, _, _) in enumerate(clips):
            print(f"  {Path(path).name:<28} " + "  ".join(f"{d['rows'][index]['wer']:7.1f}" for d in results.values()))

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
