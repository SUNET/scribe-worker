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
Speaker assignment for transcription chunks.

Pure functions over plain ``(start, end, label)`` turns, so the logic is
independent of pyannote and can be tested without models.

Speakers are assigned per word (maximum temporal overlap), and a chunk is cut
wherever the speaker changes. Assigning one speaker per chunk by looking at its
midpoint, as was done before, mislabels every chunk that contains a change of
speaker and falls back to a fixed label when the midpoint lands in a gap.
"""

from bisect import bisect_right
from typing import Optional

DEFAULT_SPEAKER = "Speaker_00"

# A run of at most this many words, shorter than MIN_RUN_SECONDS, that is
# sandwiched between two other runs is treated as diarization jitter and
# absorbed instead of producing a one word caption of its own.
MIN_RUN_WORDS = 1
MIN_RUN_SECONDS = 0.3


def normalize_speaker_name(speaker: str) -> str:
    """
    Normalize speaker names to a consistent format (Speaker_00, Speaker_01...).
    """

    if speaker.startswith("SPEAKER_"):
        return f"Speaker_{speaker.replace('SPEAKER_', '')}"

    return speaker


def turns_from_annotation(annotation) -> list:
    """
    Convert a pyannote Annotation to a start-sorted list of
    ``(start, end, speaker)`` tuples.
    """

    turns = [
        (float(segment.start), float(segment.end), normalize_speaker_name(label))
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda turn: (turn[0], turn[1]))

    return turns


class SpeakerTimeline:
    """
    Start-sorted speaker turns with overlap queries.
    """

    def __init__(self, turns: list) -> None:
        self.turns = sorted(turns, key=lambda turn: (turn[0], turn[1]))
        self._starts = [turn[0] for turn in self.turns]

    def speaker_for_span(self, start: float, end: float) -> str:
        """
        Speaker with the largest overlap with [start, end]. When nothing
        overlaps (the span falls in a gap), the nearest turn wins.
        """

        if not self.turns:
            return DEFAULT_SPEAKER

        overlap = {}
        upper = bisect_right(self._starts, end)

        for turn_start, turn_end, speaker in self.turns[:upper]:
            shared = min(end, turn_end) - max(start, turn_start)

            if shared > 0:
                overlap[speaker] = overlap.get(speaker, 0.0) + shared

        if overlap:
            return max(overlap, key=overlap.get)

        # Zero-length spans, or silence between turns: nearest turn.
        middle = (start + end) / 2
        nearest = min(
            self.turns,
            key=lambda turn: max(turn[0] - middle, middle - turn[1], 0.0),
        )

        return nearest[2]

    def speakers_in_range(self, start: float, end: float) -> list:
        """
        All speakers active at any point in [start, end], in stable order.
        """

        upper = bisect_right(self._starts, end)
        active = {
            speaker
            for turn_start, turn_end, speaker in self.turns[:upper]
            if turn_end >= start
        }

        return sorted(active)


def _runs(words: list, labels: list) -> list:
    """
    Group consecutive words with the same speaker into [speaker, [words]].
    """

    runs = []

    for word, label in zip(words, labels):
        if runs and runs[-1][0] == label:
            runs[-1][1].append(word)
        else:
            runs.append([label, [word]])

    return runs


def _absorb_jitter(runs: list) -> list:
    """
    Merge tiny interior runs into the previous run, then merge neighbours that
    now share a speaker. Runs at the edges are kept: a short "ja" at the end
    of a chunk is usually a real answer from another speaker.
    """

    index = 1

    while index < len(runs) - 1:
        run_words = runs[index][1]
        duration = run_words[-1]["e"] - run_words[0]["s"]

        if len(run_words) <= MIN_RUN_WORDS and duration < MIN_RUN_SECONDS:
            runs[index - 1][1].extend(run_words)
            del runs[index]

            # The runs on both sides of the removed one may share a speaker.
            if index < len(runs) and runs[index][0] == runs[index - 1][0]:
                runs[index - 1][1].extend(runs[index][1])
                del runs[index]

            continue

        index += 1

    return runs


def _make_segment(
    start: float,
    end: float,
    text: str,
    speaker: str,
    active: list,
    avg_score: Optional[float],
) -> dict:
    segment = {
        "start": float(start),
        "end": float(end),
        "text": text.strip(),
        "speaker": speaker,
        "active_speakers": active,
        "duration": float(end - start),
    }

    if avg_score is not None:
        segment["avg_score"] = avg_score

    return segment


def align_chunks(chunks: list, dominant: SpeakerTimeline, overlapping: SpeakerTimeline) -> list:
    """
    Turn transcription chunks into speaker-labelled segments.

    ``dominant`` decides who is speaking (use non-overlapping turns);
    ``overlapping`` is only used to report everyone active in a segment.
    Chunks carrying word timings are split at speaker changes; chunks
    without them get a single speaker chosen by maximum overlap.
    """

    segments = []

    for chunk in chunks:
        start, end, text = chunk["start"], chunk["end"], chunk["text"]
        avg_score = chunk.get("avg_score")
        words = chunk.get("words") or []

        if not words:
            segments.append(
                _make_segment(
                    start,
                    end,
                    text,
                    dominant.speaker_for_span(start, end),
                    overlapping.speakers_in_range(start, end),
                    avg_score,
                )
            )
            continue

        labels = [dominant.speaker_for_span(w["s"], w["e"]) for w in words]
        runs = _absorb_jitter(_runs(words, labels))

        if len(runs) == 1:
            segments.append(
                _make_segment(
                    start,
                    end,
                    text,
                    runs[0][0],
                    overlapping.speakers_in_range(start, end),
                    avg_score,
                )
            )
            continue

        for number, (speaker, run_words) in enumerate(runs):
            # Keep the chunk's outer bounds; pieces otherwise start and end on
            # their own first and last word, so silence between speakers stays
            # out of the captions.
            piece_start = start if number == 0 else run_words[0]["s"]
            piece_end = end if number == len(runs) - 1 else run_words[-1]["e"]
            piece_end = max(piece_end, piece_start)

            segments.append(
                _make_segment(
                    piece_start,
                    piece_end,
                    " ".join(word["t"] for word in run_words),
                    speaker,
                    overlapping.speakers_in_range(piece_start, piece_end),
                    avg_score,
                )
            )

    return segments
