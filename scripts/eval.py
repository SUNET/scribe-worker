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
The evaluation loop: measure, compare with the baseline, keep or revert.

    uv run python scripts/eval.py prepare --noise music.mp3     # once
    uv run python scripts/eval.py run                           # every change
    uv run python scripts/eval.py promote                       # accept the last run

Everything lives in one data directory (--data, default ./eval-data), per
language:

    fleurs/dev, fleurs/test   clean FLEURS clips with reference text
    dev, test                 recordings built from them (see build_noisy_set.py):
                              clean, and with noise at several SNRs, with
                              noise-only stretches between the clips
    real                      your own recordings with corrected text
    results/                  one JSON per run, and baseline.json

dev is for tuning and test is held out: look at dev while you experiment, and
let test (and real) decide. `run` compares with baseline.json and exits with
status 1 when the held-out WER, or the number of words written where nobody
speaks, gets worse by more than the limits. `promote` makes a run the new
baseline.

`run` uses the model the worker uses for the language and the decoding
settings of utils/whisper.py, so it measures what production does. Try a
change without editing the code with --set, for example --set beam_size=5
--set best_of=5, or --variant old for how it was before the fallback and
the filter.

What is measured, per condition:
    WER           substitutions, deletions and insertions over the reference
                  words. For Swedish, numbers are compared as words (see
                  evallib.normalise). "+-" is the least difference worth
                  looking at, not a significance test.
    invented      words written in the noise-only stretches, where there is
                  nothing to transcribe. This is the hallucination count.
    low/high conf the share of words that are wrong among those with a
                  confidence below 0.5 and above 0.9. The editor shows
                  confidence, so low should be much higher than high.
    x-rt          speed as a multiple of realtime.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from evallib import (
    _SV_MORPHEMES,
    VARIANTS,
    Decoder,
    default_model,
    load_audio,
    log,
    normalise,
    score,
    wer_interval,
)

FLEURS = {
    "swedish": "sv_se", "english": "en_us", "norwegian": "nb_no", "danish": "da_dk",
    "finnish": "fi_fi", "german": "de_de", "french": "fr_fr", "spanish": "es_419",
    "italian": "it_it", "russian": "ru_ru", "ukrainian": "uk_ua",
    "portuguese": "pt_br", "dutch": "nl_nl", "icelandic": "is_is",
}  # fmt: skip

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".opus", ".webm", ".mkv", ".aac"}


def git_state() -> str:
    try:
        root = Path(__file__).resolve().parent.parent
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def duration(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        ).stdout
        return float(out.strip())
    except Exception:
        return 0.0


def parse_overrides(pairs: list) -> dict:
    overrides = {}

    for pair in pairs:
        key, _, value = pair.partition("=")

        try:
            value = json.loads(value)
        except ValueError:
            pass

        overrides[key.strip()] = tuple(value) if isinstance(value, list) else value

    return overrides


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def prepare(args, root: Path) -> None:
    from build_noisy_set import build_set
    from fetch_fleurs import fetch

    if args.language not in FLEURS:
        raise SystemExit(f"no FLEURS config known for {args.language!r}: {', '.join(FLEURS)}")

    noises = args.noise or ["pink"]

    if not any(n not in ("pink", "babble") for n in noises):
        log(
            "WARNING: no --noise FILE given. Pink noise and babble are useful for WER under noise "
            "but they rarely make Whisper invent text; use a music or room recording to measure that."
        )

    for split in [s for s in args.splits.split(",") if s]:
        clips = fetch(root / "fleurs" / split, FLEURS[args.language], split, args.clips, args.seed)
        build_set(clips, root / split, noises, [float(s) for s in args.snrs.split(",")], args.gap, args.seed)

    if args.real:
        files = []

        for audio in sorted(Path(args.real).iterdir()):
            text = audio.with_suffix(".txt")

            if audio.suffix.lower() in AUDIO_SUFFIXES and text.exists():
                files.append(
                    {
                        "condition": f"real-{audio.stem}",
                        "audio": str(audio.resolve()),
                        "reference": text.read_text(encoding="utf-8").strip(),
                        "speech": [],
                        "gaps": [],
                    }
                )

        if not files:
            raise SystemExit(f"{args.real}: no audio files with a matching .txt next to them")

        (root / "real").mkdir(exist_ok=True)
        (root / "real" / "set.json").write_text(json.dumps({"files": files}, indent=1), encoding="utf-8")
        log(f"{len(files)} real recordings")

    log(f"ready: now run  eval.py run --language {args.language}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def aggregate(rows: list) -> dict:
    words = sum(r["words"] for r in rows)
    errors = sum(r["errors"] for r in rows)
    seconds = sum(r["audio_seconds"] for r in rows)
    took = sum(r["decode_seconds"] for r in rows)

    return {
        "words": words,
        "errors": errors,
        "wer": 100 * errors / max(words, 1),
        "gap_words": sum(r["gap_words"] for r in rows),
        "xrt": seconds / took if took else 0.0,
    }


def run(args, root: Path) -> int:
    splits = [s for s in args.splits.split(",") if s and (root / s / "set.json").exists()]

    if not splits:
        raise SystemExit(f"nothing to run in {root}: run  eval.py prepare  first")

    sets = {split: json.loads((root / split / "set.json").read_text(encoding="utf-8"))["files"] for split in splits}
    wanted = [c for c in args.conditions.split(",") if c]
    jobs = [
        (split, entry)
        for split in splits
        for entry in sets[split]
        if not wanted or any(w in entry["condition"] for w in wanted)
    ]

    overrides, use_filter = VARIANTS[args.variant]
    overrides = {**overrides, **parse_overrides(args.set)}
    use_filter = use_filter and not args.no_filter

    model = args.model or default_model(args.language)
    if not model:
        raise SystemExit(f"no model known for {args.language!r}: pass --model")

    log(f"loading {model} ...")
    started = time.monotonic()
    decoder = Decoder(model, args.language)
    log(f"model loaded on {decoder.device} in {time.monotonic() - started:.0f}s")
    log(f"production decoding: temperatures {decoder.temperatures}, hallucination filter "
        f"{'on' if decoder.filter_enabled else 'off'} (from utils/settings.py)")

    total = sum(duration(entry["audio"]) for _, entry in jobs)
    log(f"{len(jobs)} recordings, {total / 60:.0f} min of audio, variant '{args.variant}' {overrides or ''}"
        f"{'' if use_filter else ' without the filter'}")

    rows, confusions, done_audio, spent = [], Counter(), 0.0, 0.0

    for number, (split, entry) in enumerate(jobs, 1):
        audio = load_audio(entry["audio"])
        seconds = len(audio) / 16000
        began = time.monotonic()
        words, dropped = decoder.transcribe(audio, overrides, use_filter)
        took = time.monotonic() - began

        result = score(normalise(entry["reference"], args.language), words, entry["gaps"])
        confusions.update(result.pop("confusions"))
        result.update(
            split=split,
            condition=entry["condition"],
            dropped=len(dropped),
            audio_seconds=seconds,
            decode_seconds=took,
        )
        rows.append(result)

        done_audio += seconds
        spent += took
        eta = spent / done_audio * max(total - done_audio, 0)
        log(
            f"  [{number}/{len(jobs)}] {split}/{entry['condition']}  {seconds:.0f}s audio in {took:.0f}s  "
            f"WER {result['wer']:.1f}%  invented {result['gap_words']}  eta {eta / 60:.1f} min"
        )

    label = args.label or "-".join([args.variant] + [f"{k}={v}" for k, v in sorted(parse_overrides(args.set).items())] + (["nofilter"] if args.no_filter else []))
    output = {
        "label": label,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "model": model,
        "language": args.language,
        "device": decoder.device,
        "variant": args.variant,
        "overrides": {k: list(v) if isinstance(v, tuple) else v for k, v in overrides.items()},
        "hallucination_filter": use_filter and decoder.filter_enabled,
        "decode_temperatures": list(decoder.temperatures) if isinstance(decoder.temperatures, (list, tuple)) else [decoder.temperatures],
        "rows": rows,
        "confusions": [[ref, hyp, count] for (ref, hyp), count in confusions.most_common(300)],
    }

    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{label.replace('/', '_')[:60]}.json"
    path.write_text(json.dumps(output, indent=1, ensure_ascii=False), encoding="utf-8")

    baseline_path = results_dir / "baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() and not args.no_compare else None

    print()
    failures = report(output, baseline, args)
    error_analysis(output, args)

    print(f"\nresult: {path}")

    if baseline is None:
        print("no baseline yet: eval.py promote  makes this run the baseline")
    elif failures:
        print("\nREGRESSION:\n  " + "\n  ".join(failures))
        return 0 if args.no_gate else 1
    else:
        print("no regression against the baseline" + ("" if any(s in splits for s in args.gate_splits.split(",")) else " (dev only: not gated; test and real decide)"))

    return 0


def cell(value, width, spec=".1f", blank="-"):
    return (blank if value is None else format(value, spec)).rjust(width)


def report(output: dict, baseline, args) -> list:
    old = {(r["split"], r["condition"]): r for r in baseline["rows"]} if baseline else {}

    if baseline:
        print(f"against baseline '{baseline['label']}' ({baseline['git']}, {baseline['created']})")
        if baseline.get("model") != output["model"]:
            print(f"  note: the baseline used model {baseline.get('model')}")

    print(f"{'split':<6} {'condition':<22} {'words':>6} {'WER%':>6} {'+-':>4} {'dWER':>6} {'sub':>4} {'del':>4} {'ins':>4} "
          f"{'invented':>8} {'dInv':>5} {'lowconf':>7} {'hiconf':>6} {'x-rt':>5}")

    failures = []

    for split in dict.fromkeys(r["split"] for r in output["rows"]):
        rows = [r for r in output["rows"] if r["split"] == split]
        reference = [old[(split, r["condition"])] for r in rows if (split, r["condition"]) in old]

        for r in rows + [None]:
            if r is None:
                total = aggregate(rows)
                previous = aggregate(reference) if len(reference) == len(rows) else None
                name, words, wer = "ALL", total["words"], total["wer"]
                interval = wer_interval(total["errors"], total["words"])
                delta = wer - previous["wer"] if previous else None
                subs = sum(x["sub"] for x in rows); dels = sum(x["del"] for x in rows); inss = sum(x["ins"] for x in rows)
                invented, dinv = total["gap_words"], (total["gap_words"] - previous["gap_words"]) if previous else None
                low = high = None
                xrt = total["xrt"]

                if previous and split in args.gate_splits.split(","):
                    if delta > args.max_wer_increase:
                        failures.append(f"{split}: WER {previous['wer']:.2f}% -> {wer:.2f}% (more than +{args.max_wer_increase} points)")
                    if dinv > args.max_invented_increase:
                        failures.append(f"{split}: invented words {previous['gap_words']} -> {invented} (more than +{args.max_invented_increase})")
                    if previous["xrt"] and xrt < 0.75 * previous["xrt"]:
                        print(f"  note: {split} is {100 * (1 - xrt / previous['xrt']):.0f}% slower than the baseline")
            else:
                before = old.get((split, r["condition"]))
                name, words, wer = r["condition"], r["words"], r["wer"]
                interval = wer_interval(r["errors"], r["words"])
                delta = wer - before["wer"] if before else None
                subs, dels, inss = r["sub"], r["del"], r["ins"]
                invented = r["gap_words"]
                dinv = invented - before["gap_words"] if before else None
                low, high = r["low_conf_error_rate"], r["high_conf_error_rate"]
                xrt = r["audio_seconds"] / r["decode_seconds"] if r["decode_seconds"] else 0

            marker = "" if delta is None or abs(delta) >= interval else " ~"
            print(f"{split:<6} {name[:22]:<22} {words:>6} {wer:>6.1f} {interval:>4.1f} {cell(delta, 6, '+.1f')} {subs:>4} {dels:>4} {inss:>4} "
                  f"{invented:>8} {cell(dinv, 5, '+d')} {cell(low, 7)} {cell(high, 6)} {xrt:>5.1f}{marker}")

        print()

    print("'~' marks a change smaller than the +- interval: probably noise.")

    return failures


def error_analysis(output: dict, args) -> None:
    confusions = output["confusions"]
    numbers = set(_SV_MORPHEMES)

    def is_number(token):
        return token in numbers or token.isdigit()

    total = sum(count for _, _, count in confusions)
    numeric = sum(count for ref, hyp, count in confusions if is_number(ref) or is_number(hyp))

    print("\nwhere the errors are (all splits, the most frequent):")

    for title, keep in (
        ("substituted", lambda ref, hyp: ref and hyp),
        ("missed", lambda ref, hyp: ref and not hyp),
        ("invented", lambda ref, hyp: hyp and not ref),
    ):
        items = [(ref, hyp, count) for ref, hyp, count in confusions if keep(ref, hyp)][: args.top]

        if items:
            print(f"  {title}: " + ", ".join(
                f"{ref or '-'}>{hyp or '-'}" + (f" x{count}" if count > 1 else "") for ref, hyp, count in items
            ))

    if total:
        print(f"  {numeric} of the top {total} errors involve a number word "
              f"({100 * numeric / total:.0f}%): numbers are the place to look if that is high")

    texts = {(r["split"], r["condition"]): r["gap_text"] for r in output["rows"] if r["gap_text"]}

    if texts:
        print("\nwords written where nobody speaks:")
        for (split, condition), text in list(texts.items())[:6]:
            print(f"  {split}/{condition}: {text[:160]}")


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------


def promote(args, root: Path) -> None:
    results = root / "results"

    if args.result:
        source = Path(args.result)
    else:
        candidates = sorted(p for p in results.glob("*.json") if p.name != "baseline.json")
        if not candidates:
            raise SystemExit(f"no results in {results}")
        source = candidates[-1]

    shutil.copy(source, results / "baseline.json")
    log(f"baseline is now {source.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("command", choices=["prepare", "run", "promote"])
    parser.add_argument("result", nargs="?", default="", help="promote: the result file, default the last run")
    parser.add_argument("--data", default="eval-data")
    parser.add_argument("--language", default="swedish")
    parser.add_argument("--splits", default="dev,test,real", help="comma separated")

    prep = parser.add_argument_group("prepare")
    prep.add_argument("--clips", type=int, default=40, help="FLEURS clips per split")
    prep.add_argument("--noise", action="append", default=[], help="media file, pink or babble; repeatable")
    prep.add_argument("--snrs", default="10,0")
    prep.add_argument("--gap", type=float, default=8.0, help="noise-only seconds between clips")
    prep.add_argument("--seed", type=int, default=1)
    prep.add_argument("--real", default="", help="a folder of recordings, each with a corrected NAME.txt")

    runner = parser.add_argument_group("run")
    runner.add_argument("--model", default="", help="default: the worker's model for the language")
    runner.add_argument("--variant", default="new", choices=sorted(VARIANTS))
    runner.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a whisper-timestamped option")
    runner.add_argument("--no-filter", action="store_true", help="turn the hallucination filter off")
    runner.add_argument("--conditions", default="", help="only conditions containing one of these, comma separated")
    runner.add_argument("--label", default="")
    runner.add_argument("--no-compare", action="store_true")
    runner.add_argument("--no-gate", action="store_true", help="never exit with an error on a regression")
    runner.add_argument("--gate-splits", default="test,real", help="splits where a regression fails the run (held-out data)")
    runner.add_argument("--max-wer-increase", type=float, default=0.5, help="points, on a held-out split")
    runner.add_argument("--max-invented-increase", type=int, default=3, help="words, on a held-out split")
    runner.add_argument("--top", type=int, default=12, help="errors to list")
    args = parser.parse_args()

    root = Path(args.data) / args.language
    root.mkdir(parents=True, exist_ok=True)

    if args.command == "prepare":
        prepare(args, root)
    elif args.command == "run":
        return run(args, root)
    else:
        promote(args, root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
