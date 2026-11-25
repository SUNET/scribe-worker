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
Timing for subtitle cues.

Whisper segment boundaries are loose: they often begin inside the silence
before speech and end well after it, and a cut made mid-segment inherits the
previous half's end. Word timings are much tighter, so cues are first snapped
to them, and then adjusted to what a viewer needs: a short lead-in, enough
on-screen time to read the text, and no overlap with the next cue.
"""

from typing import Optional

from utils.settings import get_settings

settings = get_settings()

# A cue is never shorter than this, whatever the settings say.
MIN_CUE_SECONDS = 0.2


def snap_to_words(
    start: float, end: float, words: Optional[list]
) -> tuple[float, float]:
    """
    Tighten a segment's (start, end) to its first and last word.

    Falls back to the segment times when there are no words, or when the word
    times are unusable (not increasing).
    """

    if not words:
        return start, end

    word_start = words[0]["s"]
    word_end = words[-1]["e"]

    if word_end <= word_start:
        return start, end

    return word_start, word_end


def retime(cues: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """
    Adjust (start, end, caption) cues for display.

    Each cue starts slightly before its first word and stays up slightly after
    its last, is held long enough to read at SUBTITLE_MAX_CPS, and never
    overlaps its neighbours: consecutive cues are kept at least
    SUBTITLE_MIN_GAP apart.
    """

    retimed = []
    previous_end = None

    for index, (start, end, caption) in enumerate(cues):
        next_start = cues[index + 1][0] if index + 1 < len(cues) else None

        new_start = max(0.0, start - settings.SUBTITLE_LEAD_IN)

        if previous_end is not None:
            new_start = max(new_start, previous_end + settings.SUBTITLE_MIN_GAP)

        # Hold the cue long enough to read, but never past the original end
        # padded by the lead-out unless reading time demands it.
        characters = len(caption.replace("\n", " "))
        reading_time = characters / settings.SUBTITLE_MAX_CPS
        wanted = max(
            end + settings.SUBTITLE_LEAD_OUT,
            new_start + settings.SUBTITLE_MIN_DURATION,
            new_start + reading_time,
        )

        # Extra time is a courtesy and gives way to the next cue. The words
        # themselves are not shortened.
        if next_start is not None:
            wanted = min(wanted, max(end, next_start - settings.SUBTITLE_MIN_GAP))

        new_end = max(wanted, new_start + MIN_CUE_SECONDS)

        retimed.append((new_start, new_end, caption))
        previous_end = new_end

    return retimed
