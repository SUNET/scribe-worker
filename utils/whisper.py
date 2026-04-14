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

logger = get_logger()
settings = get_settings()

warnings.filterwarnings("ignore", module="pyannote")

if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN

os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

set_telemetry_metrics(False)


def get_torch_device() -> tuple:
    """
    Determine the device to use for model inference.
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
        self.__language = (
            language.split("(")[0].strip().lower() if language else language
        )
        self.__result = None
        self.__logger = logger
        self.__speakers = speakers
        self.__diarization_pipeline = diarization_object
        self.__model = load_whisper_model(model_name, logger)

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

    def __process_transcription(self, segments: list) -> dict:
        """
        Normalize and process transcription segments from whisper-timestamped.
        """
        full_transcription = ""
        processed_segments = []
        chunks = []

        for segment in segments:
            text = segment.get("text", "").strip()

            if not text:
                continue

            start_time = float(segment["start"])
            end_time = float(segment["end"])

            if full_transcription and not full_transcription.endswith(" "):
                full_transcription += " "

            full_transcription += text

            start_ms = self.__seconds_to_srt_time(start_time)
            end_ms = self.__seconds_to_srt_time(end_time)
            ts_ms = (start_ms, end_ms)
            duration = end_time - start_time

            avg_score = None
            words = segment.get("words", [])
            if words:
                confidences = [
                    float(w.get("confidence", 0))
                    for w in words
                    if w.get("text", "").strip()
                ]
                if confidences:
                    avg_score = round(float(sum(confidences) / len(confidences)), 4)

            if len(text) > 90:
                mid_index = len(text) // 2
                split_index = text.rfind(" ", 0, mid_index)

                if split_index == -1:
                    split_index = mid_index

                first_part = text[:split_index].strip()
                second_part = text[split_index:].strip()

                mid_time = start_time + duration / 2

                processed_segments.append(
                    {
                        "start": start_time,
                        "end": mid_time,
                        "text": first_part,
                        "duration": mid_time - start_time,
                        "avg_score": avg_score,
                    }
                )

                processed_segments.append(
                    {
                        "start": mid_time,
                        "end": end_time,
                        "text": second_part,
                        "duration": end_time - mid_time,
                        "avg_score": avg_score,
                    }
                )

                chunks.append(
                    {
                        "timestamp": (start_time, mid_time),
                        "timestamp_ms": (
                            ts_ms[0],
                            self.__seconds_to_srt_time(mid_time),
                        ),
                        "text": first_part,
                        "avg_score": avg_score,
                    }
                )

                chunks.append(
                    {
                        "timestamp": (mid_time, end_time),
                        "timestamp_ms": (
                            self.__seconds_to_srt_time(mid_time),
                            ts_ms[1],
                        ),
                        "text": second_part,
                        "avg_score": avg_score,
                    }
                )

                continue

            processed_segments.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "duration": duration,
                    "avg_score": avg_score,
                }
            )

            chunks.append(
                {
                    "timestamp": (start_time, end_time),
                    "timestamp_ms": ts_ms,
                    "text": text,
                    "avg_score": avg_score,
                }
            )

        converted = {
            "full_transcription": full_transcription,
            "segments": processed_segments,
            "chunks": chunks,
            "speaker_count": 1,
        }

        self.__result = converted
        self.__transcribed_seconds = (
            processed_segments[-1]["end"] if processed_segments else 0
        )

        return converted

    def __transcribe_audio(self, filepath: Optional[str] = None) -> dict:
        t0 = time.monotonic()
        if self.__audio_data:
            audio, _ = self.__decode_wav_bytes(self.__audio_data)
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
                compute_word_confidence=False,
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
                compute_word_confidence=False,
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

        return self.__process_transcription(result.get("segments", []))

    def transcribe(self) -> dict:
        """
        Transcribe the audio file and return the transcription result.
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

        if not self.__result:
            raise Exception("Transcription result is not available.")

        return self.__transcribed_seconds

    def diarization(self) -> dict:
        """
        Perform speaker diarization on the transcribed audio.
        """

        self.__logger.info(f"Starting diarization with speakers={self.__speakers}")

        t0 = time.monotonic()

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

        if not self.__result:
            raise Exception(
                "Transcription result is not available. Please transcribe first."
            )

        if self.__audio_data:
            audio_array, sample_rate = self.__decode_wav_bytes(self.__audio_data)
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

        aligned_segments = self.__align_speakers(self.__result["chunks"], diarization)

        self.__logger.info(f"Diarization completed, took {time.monotonic() - t0:.2f}s")

        return {
            "full_transcription": self.__result["full_transcription"],
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
            chunk_start = chunk["timestamp"][0]
            chunk_end = chunk["timestamp"][1]
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
        if not self.__result or "chunks" not in self.__result:
            raise Exception(
                "Transcription result is not available or does not contain chunks."
            )

        # Build list of (start_ms, end_ms, start_s, end_s, caption) entries
        entries = []
        for chunk in self.__result["chunks"]:
            text = chunk["text"].strip()
            if not text:
                continue
            start, end = chunk["timestamp_ms"]
            start_s, end_s = chunk["timestamp"]
            caption = self.__caption_split(text)
            entries.append((start, end, start_s, end_s, caption))

        # Merge consecutive single-line captions into one two-line subtitle
        # only if the gap between them is less than 1.8 seconds
        merged = []
        i = 0
        while i < len(entries):
            start, end, start_s, end_s, caption = entries[i]
            if (
                "\n" not in caption
                and i + 1 < len(entries)
                and "\n" not in entries[i + 1][4]
                and entries[i + 1][2] - end_s < 1.8
            ):
                next_start, next_end, _, _, next_caption = entries[i + 1]
                merged.append((start, next_end, f"{caption}\n{next_caption}"))
                i += 2
            else:
                merged.append((start, end, caption))
                i += 1

        subtitles = ""
        for index, (start, end, caption) in enumerate(merged):
            subtitles += f"{index + 1}\n"
            subtitles += f"{start} --> {end}\n"
            subtitles += f"{caption}\n\n"

        return subtitles

    def __caption_split(self, caption) -> str:
        """
        Split a caption into two parts if it exceeds a certain length.
        """
        if len(caption) < 42:
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
