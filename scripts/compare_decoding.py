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
Compare Whisper decoding settings on one recording, to find out which one
loses or invents speech.

    uv run python scripts/compare_decoding.py AUDIO.mp4 \\
        --model kblab/kb-whisper-large --language swedish \\
        --start 00:06:30 --end 00:07:00

Run it on the machine that has the GPU: the interesting variant (fp32) only
exists there. Each variant is decoded separately and compared with the
baseline, which uses the settings of utils/whisper.py. For every variant it
prints how many words agree with the baseline, and the stretches of three or
more words that only one of them has, with the time in the recording.

Variants (those that do not apply are skipped):
    baseline     as in production
    fp32         full precision (only differs from baseline on a GPU)
    no-vad       without the voice activity detection
    fallback     retries at higher temperatures when a decode looks wrong
    beam5        a wider beam search

There is no reference text, so a difference does not say which side is right:
listen to the stretches it lists. Words that only the baseline lacks are
speech it dropped, or that a variant invented. The transcripts are printed,
so treat the output as sensitive.
"""

import argparse
import difflib
import re
import subprocess
import sys
import time

import numpy as np

SAMPLE_RATE = 16000


def load_audio(path: str, start: str, end: str):
    """Decode a media file to 16 kHz mono float32, optionally only a stretch."""
    command = ["ffmpeg", "-v", "error"]

    if start:
        command += ["-ss", start]

    if end:
        command += ["-to", end]

    command += ["-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]

    raw = subprocess.run(command, capture_output=True, check=True).stdout

    return np.frombuffer(raw, dtype=np.float32).copy()


def seconds(text: str) -> float:
    """HH:MM:SS, MM:SS or seconds, as seconds."""
    if not text:
        return 0.0

    total = 0.0

    for part in text.split(":"):
        total = total * 60 + float(part)

    return total


def words_of(result: dict, offset: float) -> list:
    """(normalised word, absolute start time) for every word."""
    found = []

    for segment in result.get("segments", []):
        for word in segment.get("words") or []:
            text = re.sub(r"[^\w-]", "", (word.get("text") or "").lower())

            if text:
                found.append((text, word["start"] + offset))

    return found


def stamp(time_point: float) -> str:
    return f"{int(time_point // 60):02d}:{time_point % 60:05.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("audio")
    parser.add_argument("--model", default="base")
    parser.add_argument("--language", default="swedish")
    parser.add_argument("--start", default="", help="HH:MM:SS, where to start")
    parser.add_argument("--end", default="", help="HH:MM:SS, where to stop")
    parser.add_argument(
        "--variants",
        default="baseline,fp32,no-vad,fallback,beam5",
        help="comma separated, see above",
    )
    parser.add_argument("--min-words", type=int, default=3)
    args = parser.parse_args()

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

    # utils.settings parses sys.argv with the worker's own parser on import.
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        import logging

        import whisper_timestamped as whisper

        from utils.whisper import load_whisper_model
    finally:
        sys.argv = argv

    model = load_whisper_model(args.model, logging.getLogger("compare"))

    # Where the model really is: loading falls back to the CPU if the GPU
    # cannot take it.
    device = next(model.parameters()).device.type
    audio = load_audio(args.audio, args.start, args.end)
    offset = seconds(args.start)

    print(f"{len(audio) / SAMPLE_RATE:.0f} s of audio on {device}, model {args.model}\n")

    production = dict(
        vad=True,
        temperature=0.0,
        condition_on_previous_text=False,
        fp16=device != "cpu",
        compute_word_confidence=False,
        refine_whisper_precision=0,
        trust_whisper_timestamps=True,
        verbose=None,
        beam_size=3,
        best_of=3,
    )

    variants = {
        "baseline": {},
        "fp32": {"fp16": False},
        "no-vad": {"vad": False},
        "fallback": {"temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)},
        "beam5": {"beam_size": 5, "best_of": 5},
    }

    results = {}

    for name in [v for v in args.variants.split(",") if v in variants]:
        if name == "fp32" and device == "cpu":
            print(f"{name}: skipped, the CPU always uses full precision")
            continue

        settings = {**production, **variants[name]}
        started = time.monotonic()
        result = whisper.transcribe(
            model, audio, language=args.language, **settings
        )
        results[name] = words_of(result, offset)
        print(f"{name}: {len(results[name])} words, {time.monotonic() - started:.1f}s")

    if "baseline" not in results:
        return

    base = results["baseline"]

    for name, words in results.items():
        if name == "baseline":
            continue

        matcher = difflib.SequenceMatcher(
            None, [w for w, _ in base], [w for w, _ in words], autojunk=False
        )
        operations = matcher.get_opcodes()
        same = sum(i2 - i1 for kind, i1, i2, _, _ in operations if kind == "equal")

        print(
            f"\n== {name} against baseline: "
            f"{100 * same / max(len(base), len(words), 1):.0f}% of the words agree"
        )

        for kind, i1, i2, j1, j2 in operations:
            if kind == "equal" or max(i2 - i1, j2 - j1) < args.min_words:
                continue

            when = base[i1][1] if i1 < len(base) else (words[j1][1] if j1 < len(words) else 0)

            print(
                f"  {stamp(when)}  baseline: '{' '.join(w for w, _ in base[i1:i2])[:70]}'"
                f"  |  {name}: '{' '.join(w for w, _ in words[j1:j2])[:70]}'"
            )


if __name__ == "__main__":
    main()
