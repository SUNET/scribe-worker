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
import time
import torch
import warnings
import wave
import whisper_timestamped as whisper

from huggingface_hub import snapshot_download
from pyannote.audio import Pipeline
from pyannote.audio.telemetry import set_telemetry_metrics
from typing import Optional
from utils.log import get_logger
from utils.settings import get_settings
from utils.words import (
    average_confidence,
    build_payload,
    normalize_words,
    split_index_for_text,
)

logger = get_logger()
settings = get_settings()

warnings.filterwarnings("ignore", module="pyannote")

if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN

os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

set_telemetry_metrics(False)


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


def diarization_init(hf_token: str) -> Optional[Pipeline]:
    """
    Initializes the diarization pipeline using HuggingFace's PyAnnote.
    Uses the community version for better performance.
    Returns pipeline.
    """
    device, _ = get_torch_device()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=hf_token
    ).to(torch.device(device))

    return pipeline


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
        hf_token: Optional[str] = None,
        diarization_object: Optional[Pipeline] = None,
        audio_data: Optional[bytes] = None,
    ) -> None:
        self.__audio_path = audio_path
        self.__audio_data = audio_data
        self.__hf_token = hf_token
        self.__device, _ = get_torch_device()

        if "Northern Sámi" in language:
            language = "Norwegian"  # Use Norwegian model for Northern Sámi, as recommended by HuggingFace

        self.__language = (
            language.split("(")[0].strip().lower() if language else language
        )
        self.__logger = logger
        self.__speakers = speakers
        self.__diarization_pipeline = diarization_object
        self.__model = load_whisper_model(model_name, logger)
        self.__decoded_audio = None
        self.__words = []
        self.__transcribed_seconds = 0
        self.__full_transcription = ""
        # None until __process_transcription has run; an empty list is a
        # legitimate result for silent audio, so the two must stay distinct.
        self.__chunks = None

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
        seconds = float(seconds)
        millis = int(round((seconds % 1) * 1000))
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
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

            if len(text) > settings.SEGMENT_SPLIT_LENGTH:
                first_part, second_part = self.__split_text(text)
                mid_time = self.__segment_split_time(
                    start_time, end_time, words, first_part
                )

                chunks.append(
                    {
                        "start": start_time,
                        "end": mid_time,
                        "text": first_part,
                        "avg_score": avg_score,
                    }
                )

                chunks.append(
                    {
                        "start": mid_time,
                        "end": end_time,
                        "text": second_part,
                        "avg_score": avg_score,
                    }
                )

                continue

            chunks.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "avg_score": avg_score,
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
                refine_whisper_precision=0,
                trust_whisper_timestamps=True,
                verbose=False,
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
                refine_whisper_precision=0,
                trust_whisper_timestamps=True,
                verbose=False,
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

        self.__logger.info(f"Starting diarization with speakers={self.__speakers}")

        started = time.monotonic()

        speakers = int(self.__speakers)

        match speakers:
            case 0:
                min_speakers = None
                max_speakers = None
                speakers = None
            case 1:
                min_speakers = 1
                max_speakers = 2
            case _:
                min_speakers = speakers - 1
                max_speakers = speakers + 1

        if not self.__diarization_pipeline:
            self.__diarization_pipeline = diarization_init(self.__hf_token)

        if not self.__diarization_pipeline:
            raise Exception("Diarization pipeline is not available.")

        if self.__chunks is None:
            raise Exception(
                "Transcription result is not available. Please transcribe first."
            )

        if self.__audio_data:
            audio_array, sample_rate = self.__get_decoded_audio()
            waveform = torch.from_numpy(audio_array).unsqueeze(0)
            audio_input = {"waveform": waveform, "sample_rate": sample_rate}
        else:
            audio_input = self.__audio_path

        t0 = time.monotonic()
        diarization = self.__diarization_pipeline(
            audio_input,
            num_speakers=int(self.__speakers),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        self.__logger.debug(f"Diarization inference took {time.monotonic() - t0:.2f}s")

        aligned_segments = self.__align_speakers(self.__chunks, diarization)

        self.__logger.info(
            f"Diarization completed, took {time.monotonic() - started:.2f}s"
        )

        return {
            "full_transcription": self.__full_transcription,
            "segments": aligned_segments,
            "speaker_count": int(len(list(diarization.speaker_diarization.labels())))
            if diarization
            else 0,
        }

    def __align_speakers(self, transcription_chunks, diarization) -> list:
        """
        Align transcription chunks with speaker diarization results.
        """
        aligned_segments = []

        for chunk in transcription_chunks:
            chunk_start = chunk["start"]
            chunk_end = chunk["end"]
            chunk_text = chunk["text"]
            avg_score = chunk.get("avg_score")

            chunk_middle = (chunk_start + chunk_end) / 2
            dominant_speaker = self.__get_speaker(diarization, chunk_middle)
            active_speakers = self.__get_speakers_in_range(
                diarization, chunk_start, chunk_end
            )

            segment = {
                "start": float(chunk_start),
                "end": float(chunk_end),
                "text": chunk_text.strip(),
                "speaker": dominant_speaker,
                "active_speakers": active_speakers,
                "duration": float(chunk_end - chunk_start),
            }

            if avg_score is not None:
                segment["avg_score"] = avg_score

            aligned_segments.append(segment)

        return aligned_segments

    def __normalize_speaker_name(self, speaker: str) -> str:
        """
        Normalize speaker names to a consistent format (Speaker_00, Speaker_01, etc).
        """
        if speaker.startswith("SPEAKER_"):
            num = speaker.replace("SPEAKER_", "")
            return f"Speaker_{num}"
        return speaker

    def __get_speaker(self, diarization, time_point) -> str:
        """
        Get the speaker label for a specific time point in the diarization.
        """
        for segment, _, speaker in diarization.speaker_diarization.itertracks(
            yield_label=True
        ):
            if segment.start <= time_point <= segment.end:
                return self.__normalize_speaker_name(speaker)

        return "Speaker_00"

    def __get_speakers_in_range(self, diarization, start_time, end_time) -> list:
        """
        Get a list of active speakers within a specific time range in the
        diarization.
        """
        active_speakers = set()

        for segment, _, speaker in diarization.speaker_diarization.itertracks(
            yield_label=True
        ):
            if not (segment.end < start_time or segment.start > end_time):
                active_speakers.add(self.__normalize_speaker_name(speaker))

        return list(active_speakers)

    def subtitles(self) -> str:
        """
        Generate subtitles from the transcription result.
        """
        if self.__chunks is None:
            raise Exception(
                "Transcription result is not available. Please transcribe first."
            )

        # (start, end, caption), in seconds. SRT timestamps are formatted once
        # at the end, so there is only ever one representation of an instant.
        entries = [
            (chunk["start"], chunk["end"], self.__caption_split(chunk["text"].strip()))
            for chunk in self.__chunks
            if chunk["text"].strip()
        ]

        # Merge a pair of consecutive single-line captions into one two-line
        # subtitle, when the silence between them is short enough.
        merged = []
        index = 0

        while index < len(entries):
            start, end, caption = entries[index]
            following = entries[index + 1] if index + 1 < len(entries) else None

            if (
                following is not None
                and "\n" not in caption
                and "\n" not in following[2]
                and following[0] - end < settings.SUBTITLE_MERGE_GAP
            ):
                merged.append((start, following[1], f"{caption}\n{following[2]}"))
                index += 2
            else:
                merged.append((start, end, caption))
                index += 1

        return "".join(
            f"{number}\n"
            f"{self.__seconds_to_srt_time(start)} --> "
            f"{self.__seconds_to_srt_time(end)}\n"
            f"{caption}\n\n"
            for number, (start, end, caption) in enumerate(merged, start=1)
        )

    def __caption_split(self, caption) -> str:
        """
        Split a caption into two parts if it exceeds a certain length.
        """
        if len(caption) < settings.SUBTITLE_LINE_LENGTH:
            return f"{caption}"

        current_position = len(caption) // 2

        if current_position >= len(caption):
            current_position = len(caption) - 1

        characater = caption[current_position]

        while characater != " ":
            if current_position == 0 or len(caption) <= current_position:
                break

            characater = caption[current_position]
            current_position -= 1

        first_line = caption[: current_position + 1].strip()
        second_line = caption[current_position + 1 :].strip()

        if len(first_line) <= 1 or len(second_line) <= 1:
            return caption

        new_caption = f"{first_line}\n{second_line}"

        return new_caption
