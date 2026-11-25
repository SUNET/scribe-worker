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
to them (see subtitle_text), and then adjusted to what a viewer needs: a short lead-in, enough
on-screen time to read the text, and no overlap with the next cue.
"""

from typing import NamedTuple, Optional

from utils.settings import get_settings

settings = get_settings()


class Cue(NamedTuple):
    """
    One subtitle: when it is shown, what it says and, when diarization ran,
    who says it. Times are in seconds.
    """

    start: float
    end: float
    caption: str
    speaker: Optional[str] = None

# A cue is never shorter than this, whatever the settings say.
MIN_CUE_SECONDS = 0.2


def characters(caption: str) -> int:
    """
    Characters that are read, counted the way scribe-ui does: spaces and
    punctuation count, the line break does not.
    """

    return len(caption.replace("\n", ""))


def too_fast(cues: list) -> list:
    """
    The cues that are read faster than SUBTITLE_MAX_CPS. Holding a cue longer
    cannot help when the next one is in the way, or the speech is itself too
    fast, so some will remain.
    """

    return [
        cue
        for cue in cues
        if cue.end > cue.start
        and characters(cue.caption) / (cue.end - cue.start) > settings.SUBTITLE_MAX_CPS
    ]


def retime(cues: list) -> list:
    """
    Adjust cues for display.

    Each cue starts slightly before its first word and stays up slightly after
    its last, is held long enough to read at SUBTITLE_MAX_CPS, and never
    overlaps its neighbours: consecutive cues are kept at least
    SUBTITLE_MIN_GAP apart.
    """

    retimed = []
    previous_end = None

    for index, cue in enumerate(cues):
        start, end, caption = cue.start, cue.end, cue.caption
        next_start = cues[index + 1].start if index + 1 < len(cues) else None

        new_start = max(0.0, start - settings.SUBTITLE_LEAD_IN)

        if previous_end is not None:
            new_start = max(new_start, previous_end + settings.SUBTITLE_MIN_GAP)

        # Hold the cue long enough to read, but never past the original end
        # padded by the lead-out unless reading time demands it.
        reading_time = characters(caption) / settings.SUBTITLE_MAX_CPS
        wanted = max(
            end + settings.SUBTITLE_LEAD_OUT,
            new_start + settings.SUBTITLE_MIN_DURATION,
            new_start + reading_time,
        )
        wanted = min(wanted, new_start + settings.SUBTITLE_MAX_DURATION)

        # Extra time is a courtesy and gives way to the next cue. The words
        # themselves are not shortened.
        if next_start is not None:
            wanted = min(wanted, max(end, next_start - settings.SUBTITLE_MIN_GAP))

        new_end = max(wanted, new_start + MIN_CUE_SECONDS)

        retimed.append(cue._replace(start=new_start, end=new_end))
        previous_end = new_end

    return retimed
