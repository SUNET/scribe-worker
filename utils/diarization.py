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
from utils.words import average_confidence


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

    def dominant_speaker(self, start: float, end: float) -> Optional[str]:
        """
        Speaker who speaks the most of [start, end], or None if nobody does.
        Ties go to the first speaker alphabetically, so the result is stable.
        """
        overlap: dict = {}

        for i in self.__overlapping(start, end):
            shared = min(end, self.__ends[i]) - max(start, self.__starts[i])

            if shared > 0:
                speaker = self.__speakers[i]
                overlap[speaker] = overlap.get(speaker, 0.0) + shared

        if not overlap:
            return None

        return max(sorted(overlap), key=overlap.get)

    def speakers_between(self, start: float, end: float) -> list:
        """
        Sorted speakers that are active at some point in [start, end].
        """
        return sorted({self.__speakers[i] for i in self.__overlapping(start, end)})


def _word_speakers(words: list, timeline: SpeakerTimeline, fallback: str) -> list:
    """
    A speaker for each word: whoever speaks most of it. Words nobody speaks
    during (the diarization has gaps) take the speaker of the word before,
    or of the first word that has one, or the fallback.
    """
    speakers = [timeline.dominant_speaker(word["s"], word["e"]) for word in words]

    known = next((speaker for speaker in speakers if speaker), fallback)

    for i, speaker in enumerate(speakers):
        if speaker:
            known = speaker
        else:
            speakers[i] = known

    return speakers


# Whisper's word timings and the diarization's speaker changes are not exact
# to a fraction of a second, so the first or last word of a chunk, next to a
# turn change, can land on the wrong side. A single word shorter than this
# (seconds), at either end of a chunk, is not split off from its neighbour.
# A judgment call from one recording: tune it against labelled data.
MAX_EDGE_BLIP = 0.5


def _absorb_edge_blips(runs: list, words: list) -> None:
    """
    Merge a single short word at the start or end of a chunk into the run
    next to it, so that timing jitter does not split off a lone word.
    """

    def is_blip(run: list) -> bool:
        first, last = run[1], run[2]
        return last - first == 1 and (
            words[first]["e"] - words[first]["s"] < MAX_EDGE_BLIP
        )

    while len(runs) > 1 and is_blip(runs[0]):
        runs[1][1] = runs[0][1]
        del runs[0]

    while len(runs) > 1 and is_blip(runs[-1]):
        runs[-2][2] = runs[-1][2]
        del runs[-1]


def _piece(chunk: dict, timeline: SpeakerTimeline, start, end, speaker, text, score):
    piece = {
        "start": float(start),
        "end": float(end),
        "text": text.strip(),
        "speaker": speaker,
        "active_speakers": timeline.speakers_between(start, end),
        "duration": float(end - start),
    }

    if score is not None:
        piece["avg_score"] = score

    return piece


def align_speakers(chunks: list, diarization: list) -> list:
    """
    Give every transcript chunk a speaker, from (start, end, speaker) tuples.

    A chunk that carries word timings ("words", as from normalize_words) is
    decided word by word and split where the speaker changes, so a turn that
    starts in the middle of a chunk is not credited to the previous speaker.
    The pieces tile the chunk: the first starts where the chunk starts, the
    last ends where it ends, and the cut between two falls midway through the
    gap between the words on either side. A chunk without words, or one with
    a single speaker, stays whole, and then keeps its own text.

    Returns segments with start, end, text, speaker, active_speakers,
    duration and, when known, avg_score.
    """
    timeline = SpeakerTimeline(diarization)
    aligned = []

    for chunk in chunks:
        start, end = chunk["start"], chunk["end"]
        score = chunk.get("avg_score")
        words = chunk.get("words")

        fallback = timeline.speaker_at((start + end) / 2)

        if not words:
            aligned.append(
                _piece(chunk, timeline, start, end, fallback, chunk["text"], score)
            )
            continue

        speakers = _word_speakers(words, timeline, fallback)

        # Runs of consecutive words by one speaker: [speaker, first, last + 1].
        runs = []

        for i, speaker in enumerate(speakers):
            if runs and runs[-1][0] == speaker:
                runs[-1][2] = i + 1
            else:
                runs.append([speaker, i, i + 1])

        _absorb_edge_blips(runs, words)

        if len(runs) == 1:
            aligned.append(
                _piece(chunk, timeline, start, end, runs[0][0], chunk["text"], score)
            )
            continue

        # Cut midway between the last word of a run and the first of the next.
        cuts = [
            min(max((words[a[2] - 1]["e"] + words[b[1]]["s"]) / 2, start), end)
            for a, b in zip(runs, runs[1:])
        ]

        for (speaker, first, last), piece_start, piece_end in zip(
            runs, [start] + cuts, cuts + [end]
        ):
            piece_words = words[first:last]

            aligned.append(
                _piece(
                    chunk,
                    timeline,
                    piece_start,
                    piece_end,
                    speaker,
                    " ".join(word["t"] for word in piece_words),
                    average_confidence(piece_words),
                )
            )

    return aligned


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
