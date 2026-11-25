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
Subtitle cues from transcript chunks: how much text goes in a cue, and where
its line break falls.

The limits follow the Netflix timed text style guide: at most two lines of
SUBTITLE_LINE_LENGTH characters, and a cue on screen for at most
SUBTITLE_MAX_DURATION seconds.
"""

from utils.settings import get_settings
from utils.subtitle_timing import Cue
from utils.words import split_ranges

settings = get_settings()

LINES = 2

SENTENCE_END = (".", "?", "!", "…")
CLAUSE_END = (",", ";", ":", "–", "—")
CLOSING_MARKS = "\"')]}»”’"

# The words a line should not end on, and the ones a new line may start with.
# Swedish and English, which is what the workers transcribe in the main.
# Heuristics: a miss only costs a less pretty break.
NO_BREAK_AFTER = frozenset(
    "en ett den det de i på av till med för om vid från hos under över mot "
    "utan genom efter före bland mellan att som och eller men samt "
    "min mitt mina din ditt dina sin sitt sina vår vårt våra "
    "a an the of in on at to for with from by into over under as and or but "
    "my your his her its our their this that these those".split()
)
BREAK_BEFORE = frozenset(
    "och eller men samt att som för så när om fast medan eftersom "
    "and or but that which because when while if so although since".split()
)


def _bare(word: str) -> str:
    return word.lower().strip(".,;:!?…\"'()")


def wrap_caption(text: str, line_length: int) -> str:
    """
    Break a caption in two lines, if it does not fit on one.

    Both lines must fit. Among the breaks that allow it, the best is after
    punctuation or before a conjunction, and not between an article or
    preposition and the word it belongs to, nor between two names. Lines of
    equal length are preferred, with the longer one at the bottom. A text
    that cannot be broken to fit is broken as evenly as it can be.
    """

    text = " ".join(text.split())

    if len(text) <= line_length:
        return text

    words = text.split(" ")
    options = []

    for i in range(1, len(words)):
        top = " ".join(words[:i])
        bottom = " ".join(words[i:])
        last, following = words[i - 1], words[i]
        fits = len(top) <= line_length and len(bottom) <= line_length

        cost = abs(len(top) - len(bottom)) / len(text)

        if len(top) > len(bottom):
            cost += 0.1

        stripped = last.rstrip(CLOSING_MARKS)

        if stripped.endswith(SENTENCE_END):
            cost -= 0.4
        elif stripped.endswith(CLAUSE_END):
            cost -= 0.3
        elif _bare(following) in BREAK_BEFORE:
            cost -= 0.25

        if _bare(last) in NO_BREAK_AFTER and not stripped.endswith(CLAUSE_END):
            cost += 0.5

        if last[:1].isupper() and following[:1].isupper() and i > 1:
            cost += 0.4

        options.append((not fits, cost, top, bottom))

    _, _, top, bottom = min(options)

    return f"{top}\n{bottom}"


def build_cues(chunks: list) -> list:
    """
    (start, end, caption) cues from transcript chunks, in seconds.

    Times are snapped to the first and last word, since segment boundaries are
    looser than word timings. A chunk that is too long for two lines, or that
    would stay up too long, is cut where the words allow it. A cue is cut in
    proportion to the text when there are no word timings.
    """

    capacity = settings.SUBTITLE_LINE_LENGTH * LINES

    # Timing adds a lead-in and a lead-out to what is spoken.
    seconds = max(
        settings.SUBTITLE_MAX_DURATION
        - settings.SUBTITLE_LEAD_IN
        - settings.SUBTITLE_LEAD_OUT,
        settings.SUBTITLE_MIN_DURATION,
    )

    cues = []

    for chunk in chunks:
        speaker = chunk.get("speaker")
        pieces = []

        text = " ".join(chunk["text"].split())

        if not text:
            continue

        words = chunk.get("words")
        text_words = text.split(" ")

        if words and len(words) == len(text_words):
            for first, last in split_ranges(words, capacity, float("inf"), seconds):
                pieces.append(
                    (
                        *_span(chunk, words[first:last]),
                        " ".join(text_words[first:last]),
                    )
                )
        else:
            pieces = _split_plain(
                chunk["start"], chunk["end"], text, capacity, seconds
            )

        cues.extend(
            Cue(start, end, wrap_caption(piece, settings.SUBTITLE_LINE_LENGTH), speaker)
            for start, end, piece in pieces
        )

    return cues


def merge_cues(cues: list) -> list:
    """
    Put pairs of consecutive one-line cues together as one two-line cue, when
    the silence between them is short enough and the result is not on screen
    for too long.

    When two different speakers are known to say the lines, each line starts
    with SUBTITLE_SPEAKER_DASH, which counts toward its length. Two lines by
    one speaker, or by speakers that are not known, get none.
    """

    longest = (
        settings.SUBTITLE_MAX_DURATION
        - settings.SUBTITLE_LEAD_IN
        - settings.SUBTITLE_LEAD_OUT
    )
    dash = settings.SUBTITLE_SPEAKER_DASH

    merged = []
    index = 0

    while index < len(cues):
        cue = cues[index]
        following = cues[index + 1] if index + 1 < len(cues) else None

        if (
            following is not None
            and "\n" not in cue.caption
            and "\n" not in following.caption
            and following.start - cue.end < settings.SUBTITLE_MERGE_GAP
            and following.end - cue.start <= longest
        ):
            dialogue = (
                cue.speaker is not None
                and following.speaker is not None
                and cue.speaker != following.speaker
            )
            first, second = cue.caption, following.caption

            if dialogue:
                first, second = f"{dash}{first}", f"{dash}{second}"

            if max(len(first), len(second)) <= settings.SUBTITLE_LINE_LENGTH:
                merged.append(
                    Cue(
                        cue.start,
                        following.end,
                        f"{first}\n{second}",
                        None if dialogue else cue.speaker,
                    )
                )
                index += 2
                continue

        merged.append(cue)
        index += 1

    return merged


def _span(chunk: dict, words: list) -> tuple:
    """
    The (start, end) of some of a chunk's words, or of the chunk itself when
    the words say nothing sensible.
    """

    start, end = words[0]["s"], words[-1]["e"]

    if end <= start:
        return chunk["start"], chunk["end"]

    return start, end


def _split_plain(start, end, text, capacity, seconds) -> list:
    """
    Cut a cue without word timings at the space nearest the middle, as often as
    it takes, and share out the time in proportion to the text.
    """

    if (len(text) <= capacity and end - start <= seconds) or " " not in text:
        return [(start, end, text)]

    middle = len(text) // 2
    cut = text.rfind(" ", 0, middle)

    if cut <= 0:
        cut = text.find(" ", middle)

    moment = start + (end - start) * cut / len(text)

    return _split_plain(start, moment, text[:cut], capacity, seconds) + _split_plain(
        moment, end, text[cut + 1 :], capacity, seconds
    )
