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

import io
import logging
import numpy as np
import os

# Telemetry stays off. This must be set before huggingface_hub (pulled in by
# the imports below) is first imported.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import time
import torch
import wave
import whisper_timestamped as whisper

from huggingface_hub import snapshot_download
from typing import Optional
from utils.diarization import Diarizer, align_speakers, count_speakers
from utils.log import get_logger
from utils.settings import get_settings
from utils.subtitle_text import build_cues, merge_cues
from utils.subtitle_timing import retime
from utils.words import (
    average_confidence,
    build_payload,
    normalize_words,
    split_index_for_text,
    split_ranges,
)

logger = get_logger()
settings = get_settings()

# Sample rate for audio passed by path (the worker decodes to 16 kHz mono).
DIARIZATION_SAMPLE_RATE = 16000

if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN


def get_torch_device() -> tuple:
    """
    Determine the device to use for model inference.
    Returns (device, dtype).
    """
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    elif torch.backends.mps.is_available():
        return "mps", torch.float16
    else:
        return "cpu", torch.float32


def load_whisper_model(model_name: str, logger: logging.Logger) -> object:
    """
    Load a whisper model.
    """
    device, _ = get_torch_device()

    # Parse optional revision (e.g. "kblab/kb-whisper-large@strict")
    revision = None
    load_name = model_name
    if "@" in load_name:
        load_name, revision = load_name.rsplit("@", 1)

    # If a specific revision is needed, download via huggingface_hub first
    # and pass the local path to whisper-timestamped (bypasses its revision=None)
    if revision and "/" in load_name:
        load_name = snapshot_download(
            load_name,
            revision=revision,
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin"],
        )
        logger.info(f"Using '{model_name}' revision '{revision}' from {load_name}")

    try:
        model = whisper.load_model(load_name, device=device)
    except NotImplementedError:
        logger.warning(f"Failed to load model on {device}, falling back to CPU")
        device = "cpu"
        model = whisper.load_model(load_name, device=device)

    logger.info(f"Loaded model '{model_name}' on {device}")

    return model


class WhisperAudioTranscriber:
    def __init__(
        self,
        audio_path: Optional[str] = None,
        model_name: Optional[str] = "base",
        language: Optional[str] = "sv",
        speakers: Optional[int] = 0,
        audio_data: Optional[bytes] = None,
    ) -> None:
        self.__audio_path = audio_path
        self.__audio_data = audio_data
        self.__device, _ = get_torch_device()

        if "Northern Sámi" in language:
            language = "Norwegian"  # Use Norwegian model for Northern Sámi, as recommended by HuggingFace

        self.__language = (
            language.split("(")[0].strip().lower() if language else language
        )
        self.__logger = logger
        self.__speakers = speakers
        self.__diarizer = Diarizer(self.__device, logger)
        self.__model = load_whisper_model(model_name, logger)
        self.__decoded_audio = None
        self.__words = []
        self.__transcribed_seconds = 0
        self.__full_transcription = ""
        # None until __process_transcription has run; an empty list is a
        # legitimate result for silent audio, so the two must stay distinct.
        self.__chunks = None
        # Diarized pieces, with their words, once diarization() has run.
        self.__aligned = None

    def __decode_wav_bytes(self, wav_bytes: bytes) -> tuple:
        """
        Decode WAV bytes into a numpy float32 array and sample rate.
        Returns (numpy_array_float32, sample_rate).
        """
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            sample_width = wf.getsampwidth()
            raw_data = wf.readframes(n_frames)

        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        audio_array = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
        audio_array /= np.iinfo(dtype).max

        return audio_array, sample_rate

    def __get_decoded_audio(self) -> tuple:
        """
        Decode self.__audio_data once and cache the result, since both
        transcription and diarization need the same decoded array.
        """
        if self.__decoded_audio is None:
            self.__decoded_audio = self.__decode_wav_bytes(self.__audio_data)
        return self.__decoded_audio

    def __seconds_to_srt_time(self, seconds) -> str:
        """
        Convert seconds (float or string) to SRT timestamp format
        (HH:MM:SS,mmm).
        """
        # Round once, on the whole time: rounding the fraction on its own
        # can give 1000 milliseconds.
        total_millis = max(0, int(round(float(seconds) * 1000)))
        hours, remainder = divmod(total_millis, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def __segment_split_time(
        self, start_time: float, end_time: float, words: list, first_part: str
    ) -> float:
        """
        Time to cut an over-long segment at: the end of the last word that
        stays in the first half. Halving the duration would drift whenever the
        two halves are not spoken at the same pace.
        """

        word_split = split_index_for_text(words, first_part)

        if word_split is not None:
            boundary = words[word_split - 1]["e"]

            if start_time < boundary < end_time:
                return boundary

        return start_time + (end_time - start_time) / 2

    @staticmethod
    def __split_text(text: str) -> tuple:
        """
        Split an over-long segment in two, on the space nearest the middle.
        """

        mid_index = len(text) // 2
        split_index = text.rfind(" ", 0, mid_index)

        if split_index == -1:
            split_index = mid_index

        return text[:split_index].strip(), text[split_index:].strip()

    def __process_transcription(self, segments: list) -> None:
        """
        Normalize transcription segments from whisper-timestamped into the
        chunks that subtitles and speaker alignment are built from, and
        collect the per-word timings for the whole recording.
        """

        full_transcription = ""
        chunks = []
        all_words = []

        for segment in segments:
            text = segment.get("text", "").strip()

            if not text:
                continue

            start_time = float(segment["start"])
            end_time = float(segment["end"])

            if full_transcription and not full_transcription.endswith(" "):
                full_transcription += " "

            full_transcription += text

            words = normalize_words(segment.get("words"))
            all_words.extend(words)

            avg_score = average_confidence(words)

            # With word timings that match the text, cut at sentence ends,
            # clauses and long pauses, as many times as the length needs.
            text_words = text.split()

            if words and len(words) == len(text_words):
                ranges = split_ranges(
                    words,
                    settings.SEGMENT_SPLIT_LENGTH,
                    settings.SEGMENT_SPLIT_PAUSE,
                )

                if len(ranges) > 1:
                    for number, (first, last) in enumerate(ranges):
                        piece_words = words[first:last]

                        # Midway through the silence between two pieces.
                        piece_start = (
                            start_time
                            if number == 0
                            else (words[first - 1]["e"] + words[first]["s"]) / 2
                        )
                        piece_end = (
                            end_time
                            if last == len(words)
                            else (words[last - 1]["e"] + words[last]["s"]) / 2
                        )

                        # Words stamped out of order must not turn a piece
                        # inside out.
                        piece_start = min(max(piece_start, start_time), end_time)
                        piece_end = min(max(piece_end, piece_start), end_time)

                        chunks.append(
                            {
                                "start": piece_start,
                                "end": piece_end,
                                "text": " ".join(text_words[first:last]),
                                "avg_score": (
                                    average_confidence(piece_words)
                                    if any("c" in w for w in piece_words)
                                    else avg_score
                                ),
                                "words": piece_words,
                            }
                        )

                    continue

            if len(text) > settings.SEGMENT_SPLIT_LENGTH:
                first_part, second_part = self.__split_text(text)
                mid_time = self.__segment_split_time(
                    start_time, end_time, words, first_part
                )

                # The words follow the cut in time, however it was chosen.
                first_words = [w for w in words if (w["s"] + w["e"]) / 2 < mid_time]
                second_words = words[len(first_words) :]

                chunks.append(
                    {
                        "start": start_time,
                        "end": mid_time,
                        "text": first_part,
                        "avg_score": avg_score,
                        "words": first_words,
                    }
                )

                chunks.append(
                    {
                        "start": mid_time,
                        "end": end_time,
                        "text": second_part,
                        "avg_score": avg_score,
                        "words": second_words,
                    }
                )

                continue

            chunks.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "avg_score": avg_score,
                    "words": words,
                }
            )

        self.__words = all_words
        self.__full_transcription = full_transcription
        self.__chunks = chunks
        self.__transcribed_seconds = chunks[-1]["end"] if chunks else 0

    def __transcribe_audio(self, filepath: Optional[str] = None) -> None:
        t0 = time.monotonic()
        if self.__audio_data:
            audio, _ = self.__get_decoded_audio()
        else:
            audio = whisper.load_audio(filepath)
        self.__logger.debug(f"Audio decode took {time.monotonic() - t0:.2f}s")

        t0 = time.monotonic()
        try:
            result = whisper.transcribe(
                self.__model,
                audio,
                language=self.__language,
                vad=True,
                temperature=0.0,
                condition_on_previous_text=False,
                fp16=self.__device != "cpu",
                compute_word_confidence=settings.WORD_CONFIDENCE,
                refine_whisper_precision=settings.WHISPER_REFINE_PRECISION,
                trust_whisper_timestamps=True,
                verbose=None,
                beam_size=3,
                best_of=3,
            )
        except AssertionError:
            self.__logger.warning(
                "Efficient path failed, falling back to naive approach"
            )
            result = whisper.transcribe(
                self.__model,
                audio,
                language=self.__language,
                vad=True,
                temperature=0.0,
                beam_size=1,
                condition_on_previous_text=False,
                fp16=self.__device != "cpu",
                compute_word_confidence=settings.WORD_CONFIDENCE,
                refine_whisper_precision=settings.WHISPER_REFINE_PRECISION,
                trust_whisper_timestamps=True,
                verbose=None,
            )
        elapsed = time.monotonic() - t0
        audio_duration = len(audio) / 16000
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        self.__logger.info(
            f"Whisper inference took {elapsed:.2f}s for {audio_duration:.1f}s audio ({1/rtf:.1f}x realtime)"
            if rtf > 0
            else f"Whisper inference took {elapsed:.2f}s"
        )

        self.__process_transcription(result.get("segments", []))

    def transcribe(self) -> Optional[float]:
        """
        Transcribe the audio and return the number of seconds transcribed,
        or None if transcription failed.
        """
        if not self.__audio_data and self.__audio_path:
            if not os.path.exists(self.__audio_path):
                raise FileNotFoundError(
                    f"Audio file {self.__audio_path} does not exist."
                )
        elif not self.__audio_data and not self.__audio_path:
            raise ValueError("Either audio_data or audio_path must be provided.")

        try:
            self.__transcribe_audio(self.__audio_path)
        except Exception:
            self.__logger.exception("Error during transcription")
            return None

        if self.__chunks is None:
            raise Exception("Transcription result is not available.")

        return self.__transcribed_seconds

    def words(self) -> Optional[dict]:
        """
        Per-word timings and confidences for the whole recording.

        Returns None when the model produced no usable word data, so the
        caller can skip the upload rather than store an empty result.
        """

        return build_payload(self.__words)

    def diarization(self) -> dict:
        """
        Perform speaker diarization on the transcribed audio.
        """

        speakers = int(self.__speakers)

        self.__logger.info(f"Starting diarization with speakers={speakers}")

        started = time.monotonic()

        if self.__chunks is None:
            raise Exception(
                "Transcription result is not available. Please transcribe first."
            )

        if self.__audio_data:
            audio, sample_rate = self.__get_decoded_audio()
        else:
            audio, sample_rate = self.__audio_path, DIARIZATION_SAMPLE_RATE

        diarization = self.__diarizer.diarize(audio, sample_rate, speakers)

        aligned_segments = align_speakers(
            self.__chunks,
            diarization,
            audio if isinstance(audio, np.ndarray) else None,
            sample_rate,
        )

        self.__logger.info(
            f"Diarization completed, took {time.monotonic() - started:.2f}s"
        )

        self.__aligned = aligned_segments

        return {
            "full_transcription": self.__full_transcription,
            # The words are for subtitles; the upload keeps its own shape.
            "segments": [
                {key: value for key, value in segment.items() if key != "words"}
                for segment in aligned_segments
            ],
            "speaker_count": count_speakers(diarization),
        }

    def subtitles(self) -> str:
        """
        Generate subtitles from the transcription result.
        """
        if self.__chunks is None:
            raise Exception(
                "Transcription result is not available. Please transcribe first."
            )

        # Cut from the diarized pieces when there are any, so a cue never
        # runs across a change of speaker, and the dashes know who is who.
        source = (
            self.__aligned
            if self.__aligned is not None and settings.SUBTITLE_SPEAKER_DASHES
            else self.__chunks
        )

        cues = merge_cues(build_cues(source))

        return "".join(
            f"{number}\n"
            f"{self.__seconds_to_srt_time(cue.start)} --> "
            f"{self.__seconds_to_srt_time(cue.end)}\n"
            f"{cue.caption}\n\n"
            for number, cue in enumerate(retime(cues), start=1)
        )
