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
