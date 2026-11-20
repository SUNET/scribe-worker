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

        # Each speaker's segments as (start, end), sorted by start.
        self.__by_speaker: dict = {}

        for start, end, speaker in sorted(segments, key=lambda x: x[0]):
            self.__by_speaker.setdefault(speaker, []).append((start, end))

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

    def is_active(self, speaker: str, time_point: float) -> bool:
        """
        Whether the speaker is speaking at a point in time.
        """
        return any(
            self.__speakers[i] == speaker
            for i in self.__overlapping(time_point, time_point)
        )

    def next_onset(self, speaker: str, after: float, before: float) -> Optional[float]:
        """
        When the speaker next starts speaking, strictly after `after` and
        before `before`, or None.
        """
        spans = self.__by_speaker.get(speaker, [])
        i = bisect_right([start for start, _ in spans], after)

        if i < len(spans) and spans[i][0] < before:
            return spans[i][0]

        return None

    def dominant_speaker(self, start: float, end: float) -> Optional[str]:
        """
        Speaker who speaks the most of [start, end], or None if nobody does.

        When two speakers cover it about equally, one is talking over the
        other, and Whisper has transcribed one of the voices. It is the one
        whose segment goes on the longest: an interjection ends inside the main
        speaker's segment, and the previous speaker's segment ends inside the
        next speaker's turn. That speaker wins a tie, and after that the first
        alphabetically, so the result is stable.
        """
        overlap: dict = {}
        latest_end: dict = {}

        for i in self.__overlapping(start, end):
            shared = min(end, self.__ends[i]) - max(start, self.__starts[i])

            if shared > 0:
                speaker = self.__speakers[i]
                overlap[speaker] = overlap.get(speaker, 0.0) + shared
                latest_end[speaker] = max(latest_end.get(speaker, 0.0), self.__ends[i])

        if not overlap:
            return None

        best = max(overlap.values())
        tied = [speaker for speaker, shared in overlap.items() if shared >= best - TIE]

        return min(tied, key=lambda speaker: (-latest_end[speaker], speaker))

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


# Two speakers whose overlap with a word differs by less than this (seconds)
# are taken to cover it equally.
TIE = 0.02

# Audio is looked at in steps of this many seconds.
FRAME = 0.01

# A frame counts as speech when it is this many dB above the quiet floor of
# the recording (its 20th percentile), and the floor is never taken below
# FLOOR_LIMIT so that stretches of digital silence do not drag it down.
SPEECH_MARGIN_DB = 6.0

# Sound separated from the rest of a word by at least this many seconds of
# quiet is taken to belong to a neighbouring word, and so is a fragment of
# sound shorter than MIN_BLOCK (the tail of the word before, say).
MIN_PAUSE = 0.1
MIN_BLOCK = 0.06
FLOOR_LIMIT_DB = -70.0


def speech_frames(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Indices of the FRAME-long steps of the audio that hold sound, as opposed
    to the pauses between words and sentences. Steady background noise
    raises the floor with it, so it is not mistaken for speech.
    """
    hop = int(sample_rate * FRAME)
    frames = len(audio) // hop

    if frames == 0:
        return np.array([], dtype=int)

    # In blocks, so a long recording is not copied whole.
    level = np.empty(frames)
    block = 6000

    for first in range(0, frames, block):
        last = min(first + block, frames)
        part = audio[first * hop : last * hop].reshape(last - first, hop)
        rms = np.sqrt(np.mean(np.square(part, dtype=np.float32), axis=1))
        level[first:last] = 20 * np.log10(rms + 1e-9)

    floor = max(np.percentile(level, 20), FLOOR_LIMIT_DB)

    return np.flatnonzero(level > floor + SPEECH_MARGIN_DB)


def trim_to_speech(words: list, speech: np.ndarray) -> list:
    """
    Shrink each word's (s, e) to the first substantial block of sound inside
    it. Whisper stretches a word across the pause next to it, and the tail of
    the word before can reach into it, so a word that starts a turn can
    overlap the speaker before it, and one that ends a turn the speaker
    after it. Sound after the first pause belongs to the next word (Whisper
    squeezes those together), so it is not part of this one. A word with no
    sound in it is left as it was. Returns copies, in the same order.
    """
    trimmed = []
    max_step = MIN_PAUSE / FRAME

    for word in words:
        lo = int(np.searchsorted(speech, int(word["s"] / FRAME)))
        hi = int(np.searchsorted(speech, int(np.ceil(word["e"] / FRAME))))

        if hi > lo:
            # Blocks of sound, split wherever there is a real pause.
            blocks = np.split(
                speech[lo:hi], np.flatnonzero(np.diff(speech[lo:hi]) > max_step) + 1
            )
            block = next(
                (b for b in blocks if len(b) * FRAME >= MIN_BLOCK),
                max(blocks, key=len),
            )

            word = dict(word, s=block[0] * FRAME, e=(block[-1] + 1) * FRAME)

        trimmed.append(word)

    return trimmed


# When the speaker changes in the middle of a sentence, within the first or
# last word or two of it, the change is almost certainly in the wrong place:
# the diarization switched a moment too late or early (the words overlap two
# speakers about equally), or Whisper stamped those words at the wrong time.
# Up to this many words are moved, to the speaker who has the rest of the
# sentence. There is deliberately no check for a pause at the change: a
# misplaced word leaves a long fake pause behind it (over a second, on real
# recordings), so a pause says nothing about a real interruption.
MAX_SNAP_WORDS = 2

# A sentence that the previous speaker has begun is only completed by the next
# one's first words if they have said at least this many words of it.
MIN_OPEN_SENTENCE = 3

SENTENCE_END = (".", "?", "!", "\u2026")
CLOSING_MARKS = "\"')]}\u00bb\u201d\u2019"


def _ends_sentence(text: str) -> bool:
    return text.rstrip(CLOSING_MARKS).endswith(SENTENCE_END)


def _snap_to_sentence_boundary(words: list, speakers: list) -> None:
    """
    Fix a speaker change that falls in the middle of a sentence, close to
    one of its ends, by moving the few words on the short side to the speaker
    who has the rest of the sentence:

    - the change comes right after a sentence's first word or two: those
      words go to the next speaker ("... rätt. Vad | skulle det vara?");
    - the change comes right before a sentence's last word or two: those words
      go back to the previous speaker ("Borde det vara obligatoriskt
      oavsett skolans | form? Ja, men ...").

    At least one word always stays with each speaker. An interruption
    that is itself a one or two word sentence ("Jag tycker att vi | Nej!")
    looks the same as the second case and is moved too; that is rarer than
    the errors this fixes.

    words are the transcript words, speakers one entry per word, changed in
    place. The words may come from several chunks in a row, since Whisper ends
    a chunk wherever it likes, in the middle of a sentence too.
    """
    for i in range(1, len(words)):
        if speakers[i] == speakers[i - 1] or _ends_sentence(words[i - 1]["t"]):
            continue

        # Where the run before the change starts (at least one word stays).
        first_of_run = i - 1

        while first_of_run > 0 and speakers[first_of_run - 1] == speakers[i - 1]:
            first_of_run -= 1

        # Where the run after it ends (at least one word stays).
        end_of_run = i

        while end_of_run < len(words) and speakers[end_of_run] == speakers[i]:
            end_of_run += 1

        # Words of the sentence that the change interrupts, before it.
        start_of_sentence = i

        while start_of_sentence > 0 and not _ends_sentence(
            words[start_of_sentence - 1]["t"]
        ):
            start_of_sentence -= 1

        # Backwards: the sentence starts just before the change.
        back = next(
            (
                n
                for n in range(1, MAX_SNAP_WORDS + 1)
                if i - n > first_of_run and _ends_sentence(words[i - n - 1]["t"])
            ),
            None,
        )

        # Forwards: the sentence ends just after the change.
        forward = next(
            (
                n
                for n in range(1, MAX_SNAP_WORDS + 1)
                if i + n < end_of_run
                and i - start_of_sentence >= MIN_OPEN_SENTENCE
                and _ends_sentence(words[i + n - 1]["t"])
            ),
            None,
        )

        # A sentence the change interrupts is at least MIN_OPEN_SENTENCE words
        # long before it, and more than MAX_SNAP_WORDS, so at most one of the
        # two can apply.
        if back is not None:
            for k in range(i - back, i):
                speakers[k] = speakers[i]
        elif forward is not None:
            for k in range(i, i + forward):
                speakers[k] = speakers[i - 1]


# A piece should not start long before its speaker does. Diarization onsets
# lag a little behind the speech (by a tenth of a second or so), so a start
# that precedes the onset by no more than this is left as Whisper had it.
ONSET_TOLERANCE = 0.15


def _start_at_onset(
    timeline: "SpeakerTimeline",
    speech: Optional[np.ndarray],
    speaker: str,
    start: float,
    end: float,
) -> float:
    """
    Where a piece by the speaker should start, given that Whisper started it
    at `start`. Whisper stamps words wrongly when its text skips something
    that was said, and then a piece can start seconds before its speaker has
    said anything. If the speaker is not speaking at `start` and does not
    begin to within ONSET_TOLERANCE, it starts at the first sound from the
    diarization onset (less the tolerance) instead, as long as that is still
    inside the piece.
    """
    if timeline.is_active(speaker, start):
        return start

    onset = timeline.next_onset(speaker, start, end)

    if onset is None or onset - start <= ONSET_TOLERANCE:
        return start

    earliest = onset - ONSET_TOLERANCE

    if speech is not None:
        i = int(np.searchsorted(speech, int(np.ceil(earliest / FRAME))))

        if i < len(speech) and speech[i] * FRAME < end:
            return max(start, float(speech[i] * FRAME))

    return max(start, earliest)


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


def align_speakers(
    chunks: list,
    diarization: list,
    audio: Optional[np.ndarray] = None,
    sample_rate: int = nemotron.SAMPLE_RATE,
) -> list:
    """
    Give every transcript chunk a speaker, from (start, end, speaker) tuples.

    A chunk that carries word timings ("words", as from normalize_words) is
    decided word by word and split where the speaker changes, so a turn that
    starts in the middle of a chunk is not credited to the previous speaker.
    The pieces tile the chunk: the first starts where the chunk starts, the
    last ends where it ends, and the cut between two falls midway through the
    gap between the words on either side. A chunk without words, or one with
    a single speaker, stays whole, and then keeps its own text.

    With the audio (a float32 mono array), words are first trimmed to where
    there is sound (see trim_to_speech), which keeps a word next to a pause
    from being credited to the wrong side of a turn change. A change of
    speaker in the middle of a sentence, within its first or last word or two,
    is moved to the sentence's edge (see _snap_to_sentence_boundary), also
    across the boundary between two chunks.

    Returns segments with start, end, text, speaker, active_speakers,
    duration and, when known, avg_score.
    """
    timeline = SpeakerTimeline(diarization)
    speech = speech_frames(audio, sample_rate) if audio is not None else None

    # Per chunk with words: the words trimmed to speech, and who speaks each.
    # Trimmed words decide who speaks and where the cuts fall; the original
    # words supply the text.
    prepared = []

    for chunk in chunks:
        words = chunk.get("words")

        if not words:
            prepared.append(None)
            continue

        heard = trim_to_speech(words, speech) if speech is not None else words
        fallback = timeline.speaker_at((chunk["start"] + chunk["end"]) / 2)
        prepared.append((heard, _word_speakers(heard, timeline, fallback)))

    # Speaker changes are fixed over runs of chunks that have words, so a
    # change at the very end of one chunk can still be moved.
    index = 0

    while index < len(chunks):
        if prepared[index] is None:
            index += 1
            continue

        end = index

        while end < len(chunks) and prepared[end] is not None:
            end += 1

        group = range(index, end)
        flat_words = [word for i in group for word in chunks[i]["words"]]
        flat_speakers = [speaker for i in group for speaker in prepared[i][1]]

        _snap_to_sentence_boundary(flat_words, flat_speakers)

        position = 0

        for i in group:
            count = len(prepared[i][1])
            prepared[i] = (prepared[i][0], flat_speakers[position : position + count])
            position += count

        index = end

    aligned = []

    for chunk, ready in zip(chunks, prepared):
        start, end = chunk["start"], chunk["end"]
        score = chunk.get("avg_score")

        if ready is None:
            speaker = timeline.speaker_at((start + end) / 2)
            aligned.append(
                _piece(
                    chunk,
                    timeline,
                    _start_at_onset(timeline, speech, speaker, start, end),
                    end,
                    speaker,
                    chunk["text"],
                    score,
                )
            )
            continue

        words = chunk["words"]
        heard, speakers = ready

        # Runs of consecutive words by one speaker: [speaker, first, last + 1].
        runs = []

        for i, speaker in enumerate(speakers):
            if runs and runs[-1][0] == speaker:
                runs[-1][2] = i + 1
            else:
                runs.append([speaker, i, i + 1])

        if len(runs) == 1:
            speaker = runs[0][0]
            aligned.append(
                _piece(
                    chunk,
                    timeline,
                    _start_at_onset(timeline, speech, speaker, start, end),
                    end,
                    speaker,
                    chunk["text"],
                    score,
                )
            )
            continue

        # Cut midway between the last word of a run and the first of the next,
        # but not before the next speaker has begun to speak.
        cuts = [
            min(max((heard[a[2] - 1]["e"] + heard[b[1]]["s"]) / 2, start), end)
            for a, b in zip(runs, runs[1:])
        ]

        for k in range(len(cuts)):
            limit = cuts[k + 1] if k + 1 < len(cuts) else end
            cuts[k] = _start_at_onset(timeline, speech, runs[k + 1][0], cuts[k], limit)

        starts = [_start_at_onset(timeline, speech, runs[0][0], start, cuts[0])]

        for (speaker, first, last), piece_start, piece_end in zip(
            runs, starts + cuts, cuts + [end]
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
