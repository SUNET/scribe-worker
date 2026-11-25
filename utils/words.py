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
Word level timing payload ("words" result format).

The payload is a flat, time ordered list of every spoken word with its
absolute start/end time and, when available, a confidence score. It is
deliberately *not* nested inside the segment structure: consumers map words
onto captions by time range, so the payload stays valid after a user splits,
merges or re-times captions in the editor.

Keys are short because the payload is stored encrypted as a single blob and
a one hour transcription contains roughly ten thousand words:

    {
        "version": 1,
        "words": [{"t": "Hej", "s": 0.12, "e": 0.34, "c": 0.98}, ...]
    }

    t  text        the word as transcribed, whitespace stripped
    s  start       absolute start time in seconds
    e  end         absolute end time in seconds
    c  confidence  0.0-1.0, omitted entirely when the model did not report it

This format is additive. Older consumers never request it and are unaffected.
"""

from typing import Any, Optional

# Bump when the payload shape changes incompatibly. Consumers must treat an
# unknown version as "no word data" rather than guessing. Deliberately not a
# setting: a deployment claiming to speak a version the code does not
# implement would only mis-parse silently.
WORDS_FORMAT_VERSION = 1

# Timestamps are rounded to milliseconds and confidences to three decimals.
# Anything finer is noise from the alignment model and only costs bytes.
TIME_PRECISION = 3
CONFIDENCE_PRECISION = 3


def normalize_word(word: Any) -> Optional[dict]:
    """
    Convert a single whisper-timestamped word into the compact payload form.

    Returns None for words that carry no usable text or timing.
    """

    if not isinstance(word, dict):
        return None

    # whisper-timestamped uses "text"; some versions/forks use "word".
    text = (word.get("text") or word.get("word") or "").strip()

    if not text:
        return None

    start = word.get("start")
    end = word.get("end")

    if start is None or end is None:
        return None

    try:
        entry = {
            "t": text,
            "s": round(float(start), TIME_PRECISION),
            "e": round(float(end), TIME_PRECISION),
        }
    except (TypeError, ValueError):
        return None

    confidence = word.get("confidence")

    if confidence is not None:
        try:
            entry["c"] = round(float(confidence), CONFIDENCE_PRECISION)
        except (TypeError, ValueError):
            pass

    return entry


def normalize_words(words: Optional[list]) -> list:
    """
    Normalize a list of whisper-timestamped words, dropping unusable entries.
    """

    if not words:
        return []

    return [entry for entry in (normalize_word(word) for word in words) if entry]


def average_confidence(words: Optional[list]) -> Optional[float]:
    """
    Average confidence over normalized words.

    Returns None when no word reported a confidence, so that "confidence was
    not computed" stays distinguishable from "the model was not confident".
    """

    if not words:
        return None

    scores = [word["c"] for word in words if "c" in word]

    if not scores:
        return None

    return round(sum(scores) / len(scores), CONFIDENCE_PRECISION)


def build_payload(words: list) -> Optional[dict]:
    """
    Wrap normalized words in the versioned envelope.

    Returns None when there is nothing to send, so callers can skip the
    upload entirely instead of storing an empty blob.
    """

    if not words:
        return None

    return {"version": WORDS_FORMAT_VERSION, "words": words}


def split_index_for_text(words: list, first_part: str) -> Optional[int]:
    """
    Number of words consumed by ``first_part`` when a segment is split in two.

    Used to pick a real word boundary for the new timestamp instead of
    halving the segment duration. Returns None when the split does not land
    strictly inside the word list, in which case the caller should fall back
    to its own estimate.
    """

    if not words:
        return None

    count = len(first_part.split())

    if count <= 0 or count >= len(words):
        return None

    return count


# What a split point is worth, against how far it is from the middle of the
# segment (0 on the middle, 0.5 at either end). A sentence end beats a comma,
# and a pause beats both once it is long enough.
SENTENCE_BONUS = 0.3
CLAUSE_BONUS = 0.15
PAUSE_BONUS_PER_SECOND = 0.3
PAUSE_BONUS_MAX = 0.3
CLAUSE_MARKS = (",", ";", ":", "–", "—")
SENTENCE_MARKS = (".", "?", "!", "…")

# Neither half of a split may be shorter than this share of the text.
MIN_SPLIT_SHARE = 0.25


def _ends_with(text: str, marks: tuple) -> bool:
    return text.rstrip("\"')]}»”’").endswith(marks)


def split_ranges(
    words: list,
    max_chars: int,
    pause: float,
    max_seconds: float = float("inf"),
    _lo: int = 0,
    _hi: Optional[int] = None,
) -> list:
    """
    Cut a segment's words into (first, last + 1) index ranges, for captions.

    A pause of at least ``pause`` seconds between two words is always a cut.
    A piece still longer than ``max_chars`` is cut in two at the best word
    boundary near its middle: one after a sentence end, then a comma, then a
    longer pause, over a cut that merely lands on the middle. Pieces are cut
    again until they fit, or have no boundary left to cut at. ``max_seconds``
    limits a piece the same way, from its first word to its last.
    """

    hi = len(words) if _hi is None else _hi

    # Long pauses first, since they are real breaks in the speech.
    cuts = [
        i for i in range(_lo + 1, hi) if words[i]["s"] - words[i - 1]["e"] >= pause
    ]

    if cuts:
        bounds = [_lo] + cuts + [hi]
        ranges = []

        for a, b in zip(bounds, bounds[1:]):
            ranges.extend(
                split_ranges(words, max_chars, float("inf"), max_seconds, a, b)
            )

        return ranges

    sizes = [len(word["t"]) + 1 for word in words[_lo:hi]]
    total = sum(sizes)

    seconds = words[hi - 1]["e"] - words[_lo]["s"] if hi > _lo else 0.0

    if (total - 1 <= max_chars and seconds <= max_seconds) or hi - _lo < 2:
        return [(_lo, hi)]

    best = None
    before = 0

    for i in range(_lo + 1, hi):
        before += sizes[i - 1 - _lo]
        share = before / total

        if not MIN_SPLIT_SHARE <= share <= 1 - MIN_SPLIT_SHARE:
            continue

        text = words[i - 1]["t"]
        cost = abs(share - 0.5)

        if _ends_with(text, SENTENCE_MARKS):
            cost -= SENTENCE_BONUS
        elif _ends_with(text, CLAUSE_MARKS):
            cost -= CLAUSE_BONUS

        gap = max(0.0, words[i]["s"] - words[i - 1]["e"])
        cost -= min(gap * PAUSE_BONUS_PER_SECOND, PAUSE_BONUS_MAX)

        if best is None or cost < best[0]:
            best = (cost, i)

    if best is None:
        return [(_lo, hi)]

    # A pause was already handled above, so recursion here only has the
    # length to deal with.
    return split_ranges(
        words, max_chars, float("inf"), max_seconds, _lo, best[1]
    ) + split_ranges(words, max_chars, float("inf"), max_seconds, best[1], hi)
