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

"""
Speaker diarization: picks the engine and holds the loaded models.

Nemotron is used unless more speakers are configured than it handles. If it
reports its maximum, the count may have been capped (it cannot say "more"),
so the audio is diarized again with pyannote. pyannote's result is used only
if it finds more speakers than Nemotron's maximum; otherwise the capping did
not matter and Nemotron's result, kept from the first run, is used.
"""

import logging
import time

from bisect import bisect_right
from itertools import accumulate

from typing import Any, Optional, Union

import numpy as np

from utils import nemotron, pyannote


def count_speakers(segments: list) -> int:
    """
    Number of distinct speakers in (start, end, speaker) tuples.
    """
    return len({speaker for _, _, speaker in segments})


class SpeakerTimeline:
    """
    Who spoke when, indexed for fast lookups. Built once from (start, end,
    speaker) tuples, in any order and with overlaps allowed, then queried for
    each transcript chunk without scanning every segment.
    """

    def __init__(self, segments: list) -> None:
        # Sorted by start. Remembering the original position keeps the
        # "first segment in the list wins" rule of speaker_at().
        order = sorted(range(len(segments)), key=lambda i: segments[i][0])

        self.__positions = order
        self.__starts = [segments[i][0] for i in order]
        self.__ends = [segments[i][1] for i in order]
        self.__speakers = [segments[i][2] for i in order]

        # Running maximum of the ends: once it drops below the time asked
        # for, no earlier segment can contain it.
        self.__max_ends = list(accumulate(self.__ends, max))

    def __overlapping(self, start: float, end: float):
        """
        Yield the sorted index of every segment that overlaps [start, end].
        """
        i = bisect_right(self.__starts, end) - 1

        while i >= 0 and self.__max_ends[i] >= start:
            if self.__ends[i] >= start:
                yield i

            i -= 1

    def speaker_at(self, time_point: float) -> str:
        """
        Speaker at a point in time. If several overlap, the one that comes
        first in the original list wins. Speaker_00 if nobody speaks.
        """
        found = min(
            self.__overlapping(time_point, time_point),
            key=lambda i: self.__positions[i],
            default=None,
        )

        return "Speaker_00" if found is None else self.__speakers[found]

    def speakers_between(self, start: float, end: float) -> list:
        """
        Sorted speakers that are active at some point in [start, end].
        """
        return sorted({self.__speakers[i] for i in self.__overlapping(start, end)})


class Diarizer:
    """
    Diarizes audio, loading each engine's model on first use and keeping it
    for later calls.
    """

    def __init__(self, device: str, logger: logging.Logger) -> None:
        self.__device = device
        self.__logger = logger
        self.__nemotron: Optional[Any] = None
        self.__pyannote: Optional[Any] = None

    def diarize(
        self,
        audio: Union[np.ndarray, str],
        sample_rate: int,
        speakers: int = 0,
    ) -> list:
        """
        Diarize a float32 mono array (at sample_rate) or an audio file path.

        speakers is the configured number of speakers (0 = detect). Nemotron
        detects the number itself and ignores it, so it only decides the
        engine and is passed on to pyannote as a hint. Returns (start, end,
        speaker) tuples, with speakers named Speaker_00, Speaker_01, ...
        """
        nemotron_result = None

        if speakers <= nemotron.MAX_SPEAKERS:
            nemotron_result = self.__run_nemotron(audio, sample_rate)
            found = count_speakers(nemotron_result)

            if found < nemotron.MAX_SPEAKERS:
                return nemotron_result

            self.__logger.info(
                f"Nemotron reached its limit of {nemotron.MAX_SPEAKERS} speakers, "
                "so the count may be capped. Diarizing again with pyannote."
            )
        else:
            self.__logger.info(
                f"{speakers} speakers configured, more than Nemotron's "
                f"{nemotron.MAX_SPEAKERS}. Skipping it."
            )

        result = self.__run_pyannote(audio, sample_rate, speakers)

        if nemotron_result is not None:
            found = count_speakers(result)

            if found <= nemotron.MAX_SPEAKERS:
                self.__logger.info(
                    f"pyannote did not find more than {nemotron.MAX_SPEAKERS} "
                    "speakers. Using the Nemotron result."
                )
                return nemotron_result

        return result

    def __run_nemotron(self, audio, sample_rate: int) -> list:
        self.__logger.info("Diarizing with Nemotron")

        if not self.__nemotron:
            self.__nemotron = nemotron.load_model()

        t0 = time.monotonic()
        segments = nemotron.diarize(self.__nemotron, audio, sample_rate)
        self.__logger.info(
            f"Nemotron found {count_speakers(segments)} speakers, "
            f"took {time.monotonic() - t0:.2f}s"
        )

        return segments

    def __run_pyannote(self, audio, sample_rate: int, speakers: int) -> list:
        self.__logger.info("Diarizing with pyannote")

        if not self.__pyannote:
            self.__pyannote = pyannote.load_pipeline(self.__device)

        t0 = time.monotonic()
        segments = pyannote.diarize(self.__pyannote, audio, speakers, sample_rate)
        self.__logger.info(
            f"pyannote found {count_speakers(segments)} speakers, "
            f"took {time.monotonic() - t0:.2f}s"
        )

        return segments
