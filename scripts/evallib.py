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
Shared pieces of the evaluation scripts: text normalisation, word alignment
and scoring, audio helpers, and a Decoder that transcribes the way the worker
does. Importing this module is cheap; torch and Whisper load only when a
Decoder is made.
"""

import difflib
import math
import re
import subprocess
import sys
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_SV_UNITS = [
    "noll", "ett", "två", "tre", "fyra", "fem", "sex", "sju", "åtta", "nio",
    "tio", "elva", "tolv", "tretton", "fjorton", "femton", "sexton", "sjutton",
    "arton", "nitton",
]  # fmt: skip
_SV_TENS = {
    2: "tjugo", 3: "trettio", 4: "fyrtio", 5: "femtio",
    6: "sextio", 7: "sjuttio", 8: "åttio", 9: "nittio",
}  # fmt: skip
_SV_MORPHEMES = sorted(
    set(_SV_UNITS) | set(_SV_TENS.values()) | {"en", "hundra", "tusen"},
    key=len,
    reverse=True,
)


def swedish_number_words(number: int) -> Optional[list]:
    """
    A number below a million as Swedish words, one token per part, so that
    "2019" gives ["två", "tusen", "nitton"]. A leading "ett" before hundra or
    tusen is left out, as normalise() does on both sides.
    """

    if number < 0 or number >= 1_000_000:
        return None

    if number < 20:
        return [_SV_UNITS[number]]

    if number < 100:
        words = [_SV_TENS[number // 10]]
        return words + (swedish_number_words(number % 10) if number % 10 else [])

    if number < 1000:
        words = ([_SV_UNITS[number // 100]] if number >= 200 else []) + ["hundra"]
        return words + (swedish_number_words(number % 100) if number % 100 else [])

    head = number // 1000
    words = (swedish_number_words(head) if head > 1 else []) + ["tusen"]
    return words + (swedish_number_words(number % 1000) if number % 1000 else [])


def _segment_number_word(token: str) -> Optional[list]:
    """
    Split a compound such as "tvåhundratjugotre" into number words, or None
    when the token is not made up of number words alone (or is only one).
    """

    if len(token) < 5:
        return None

    parts, rest = [], token

    while rest:
        for morpheme in _SV_MORPHEMES:
            if rest.startswith(morpheme):
                parts.append(morpheme)
                rest = rest[len(morpheme):]
                break
        else:
            return None

    return parts if len(parts) > 1 else None


def _swedish_numbers(tokens: list) -> list:
    expanded = []

    for token in tokens:
        if token.isdigit() and not (len(token) > 1 and token.startswith("0")):
            words = swedish_number_words(int(token))
            expanded.extend(words if words else [token])
            continue

        expanded.extend(_segment_number_word(token) or [token])

    # "ett hundra" and "hundra" are the same number.
    return [
        token
        for index, token in enumerate(expanded)
        if not (token == "ett" and index + 1 < len(expanded) and expanded[index + 1] in ("hundra", "tusen"))
    ]


def normalise(text: str, language: Optional[str] = None) -> list:
    """
    Words of a text for comparison: lower case, no punctuation. For Swedish,
    digits and compound number words are written out as separate number
    words, so "2019", "tjugohundranitton" and "två tusen nitton" are compared
    on their words, not on how they happen to be spelled. (A year said in
    the "tjugo nitton" way still differs from "2019": that cannot be told
    from the text.)
    """

    tokens = re.sub(r"[^\w\s]", " ", text.lower().replace("-", " ")).split()

    if language == "swedish":
        tokens = _swedish_numbers(tokens)

    return tokens


# ---------------------------------------------------------------------------
# Alignment and scoring
# ---------------------------------------------------------------------------


def align_ops(reference: list, hypothesis: list) -> list:
    """
    A minimal word edit script as (operation, reference index, hypothesis
    index) triples, where the operation is eq, sub, del or ins.
    """

    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j

    for i in range(1, n + 1):
        row, above = cost[i], cost[i - 1]
        ref_word = reference[i - 1]

        for j in range(1, m + 1):
            row[j] = min(
                above[j - 1] + (ref_word != hypothesis[j - 1]),
                above[j] + 1,
                row[j - 1] + 1,
            )

    ops, i, j = [], n, m

    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            ops.append(("eq" if reference[i - 1] == hypothesis[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("del", i - 1, j))
            i -= 1
        else:
            ops.append(("ins", i, j - 1))
            j -= 1

    return ops[::-1]


def align(reference: list, hypothesis: list) -> tuple:
    """Substitutions, deletions and insertions."""
    counts = Counter(op for op, _, _ in align_ops(reference, hypothesis))
    return counts["sub"], counts["del"], counts["ins"]


def describe(reference: list, hypothesis: list) -> str:
    """What differs, as '[ref > hyp]' pairs, '-' for a missing word."""
    parts = []
    matcher = difflib.SequenceMatcher(None, reference, hypothesis, autojunk=False)

    for kind, i1, i2, j1, j2 in matcher.get_opcodes():
        if kind != "equal":
            parts.append(f"[{' '.join(reference[i1:i2]) or '-'} > {' '.join(hypothesis[j1:j2]) or '-'}]")

    return " ".join(parts)


def wer_interval(errors: int, words: int) -> float:
    """
    Half width, in percentage points, of a rough 95% interval around a WER.
    It treats every word as an independent trial, which it is not, so read it
    as the least difference worth looking at, not as a test.
    """

    if words <= 0:
        return 0.0

    p = min(max(errors / words, 1e-9), 1 - 1e-9)
    return 196 * math.sqrt(p * (1 - p) / words)


def score(reference: list, hyp_words: list, gaps: list, margin: float = 1.0) -> dict:
    """
    Score hypothesis words, dicts with t (token), s and e (seconds) and
    optionally c (confidence), against reference tokens.

    gaps are (start, end) stretches that hold no speech. A hypothesis word
    whose middle falls more than `margin` seconds inside one is counted as
    invented: word timings drift a little into the neighbouring speech.
    """

    tokens = [w["t"] for w in hyp_words]
    ops = align_ops(reference, tokens)
    counts = Counter(op for op, _, _ in ops)

    inner = [(a + margin, b - margin) for a, b in gaps if b - a > 2 * margin]
    invented = [
        w["t"] for w in hyp_words if any(a <= (w["s"] + w["e"]) / 2 <= b for a, b in inner)
    ]

    confusions = Counter()
    right, wrong = [], []

    for op, i, j in ops:
        if op == "sub":
            confusions[(reference[i], tokens[j])] += 1
        elif op == "del":
            confusions[(reference[i], "")] += 1
        elif op == "ins":
            confusions[("", tokens[j])] += 1

        if op in ("eq", "sub", "ins") and "c" in hyp_words[j]:
            (right if op == "eq" else wrong).append(hyp_words[j]["c"])

    def error_rate(low: float, high: float) -> Optional[float]:
        bucket = [(c, True) for c in right] + [(c, False) for c in wrong]
        bucket = [ok for c, ok in bucket if low <= c < high]
        return round(100 * bucket.count(False) / len(bucket), 1) if len(bucket) >= 5 else None

    errors = counts["sub"] + counts["del"] + counts["ins"]

    return {
        "words": len(reference),
        "sub": counts["sub"],
        "del": counts["del"],
        "ins": counts["ins"],
        "errors": errors,
        "wer": 100 * errors / max(len(reference), 1),
        "gap_words": len(invented),
        "gap_text": " ".join(invented),
        "gap_seconds": sum(max(b - a, 0) for a, b in inner),
        "low_conf_error_rate": error_rate(0.0, 0.5),
        "high_conf_error_rate": error_rate(0.9, 1.01),
        "confusions": confusions,
    }


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def load_audio(path: str) -> np.ndarray:
    """Any media file as 16 kHz mono float32."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True,
        check=True,
    ).stdout

    return np.frombuffer(raw, dtype=np.float32).copy()


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {message}", flush=True)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

# name: (whisper-timestamped overrides of the production call, hallucination filter)
VARIANTS = {
    "old": ({"temperature": 0.0}, False),
    "new": ({}, True),
    "novad": ({"vad": False}, True),
    "beam5": ({"beam_size": 5, "best_of": 5}, True),
}


def default_model(language: str) -> Optional[str]:
    """The model the worker uses for a language, from utils/settings.py."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    argv, sys.argv = sys.argv, sys.argv[:1]

    try:
        from utils.settings import get_settings

        for name, models in get_settings().WHISPER_MODELS_HF.items():
            if name.lower() == language.lower() and models:
                return next(iter(models.values()))
    finally:
        sys.argv = argv

    return None


class Decoder:
    """Transcribes with the settings of utils/whisper.py."""

    def __init__(self, model_name: str, language: str) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

        # utils.settings parses sys.argv with the worker's own parser on import.
        argv, sys.argv = sys.argv, sys.argv[:1]
        try:
            import logging

            import whisper_timestamped

            from utils.hallucination import is_hallucination
            from utils.settings import get_settings
            from utils.whisper import load_whisper_model
            from utils.words import average_confidence, normalize_words
        finally:
            sys.argv = argv

        self._whisper = whisper_timestamped
        self._is_hallucination = is_hallucination
        self._average_confidence = average_confidence
        self._normalize_words = normalize_words
        self.settings = get_settings()
        # Older checkouts of the worker have neither setting: then production
        # decodes at temperature 0.0 and does not filter, and so do we.
        self.temperatures = getattr(self.settings, "DECODE_TEMPERATURES", 0.0)
        self.filter_enabled = getattr(self.settings, "HALLUCINATION_FILTER", False)
        self.language = language
        self.model_name = model_name

        self.model = load_whisper_model(model_name, logging.getLogger("eval"))
        # Where the model really is: loading falls back to the CPU.
        self.device = next(self.model.parameters()).device.type

        self.production = dict(
            vad=True,
            temperature=self.temperatures,
            condition_on_previous_text=False,
            fp16=self.device != "cpu",
            compute_word_confidence=True,
            refine_whisper_precision=0,
            trust_whisper_timestamps=True,
            verbose=None,
            beam_size=3,
            best_of=3,
        )

    def transcribe(self, audio: np.ndarray, overrides: dict, use_filter: bool) -> tuple:
        """
        Words as dicts (t, s, e, c) from the segments that survive the
        hallucination filter, and the texts of those it dropped.
        """

        result = self._whisper.transcribe(
            self.model, audio, language=self.language, **{**self.production, **overrides}
        )

        words, dropped = [], []

        for segment in result.get("segments", []):
            text = (segment.get("text") or "").strip()
            segment_words = self._normalize_words(segment.get("words"))

            if (
                use_filter
                and self.filter_enabled
                and self._is_hallucination(text, self._average_confidence(segment_words), self.language)
            ):
                dropped.append(text)
                continue

            for word in segment_words:
                for token in normalise(word["t"], self.language):
                    entry = {"t": token, "s": word["s"], "e": word["e"]}

                    if "c" in word:
                        entry["c"] = word["c"]

                    words.append(entry)

        return words, dropped
