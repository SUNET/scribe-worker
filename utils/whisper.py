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
import json
import logging
import numpy as np
import os
import re
import subprocess
import tempfile
import torch
import wave

from pyannote.audio import Pipeline
from pyannote.audio.telemetry import set_telemetry_metrics
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Optional
from utils.settings import get_settings

settings = get_settings()

os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
set_telemetry_metrics(False, save_choice_as_default=True)

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
    """
    device, _ = get_torch_device()

    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=hf_token
    ).to(torch.device(device))


class WhisperAudioTranscriber:
    def __init__(
        self,
        logger: logging.Logger,
        backend: str,
        audio_path: Optional[str] = None,
        model_name: Optional[str] = "KBLab/kb-whisper-base",
        language: Optional[str] = "sv",
        speakers: Optional[int] = 0,
        hf_token: Optional[str] = None,
        whisper_cpp_path: Optional[str] = settings.WHISPER_CPP_PATH,
        diarization_object: Optional[Pipeline] = None,
        audio_data: Optional[bytes] = None,
    ) -> None:
        """
        Initializes the WhisperAudioTranscriber with the audio
        file path, model name,
        """

        self.__audio_path = audio_path
        self.__audio_data = audio_data
        self.__model_name = model_name
        self.__hf_token = hf_token
        self.__device, self.__torch_dtype = get_torch_device()
        self.__language = language
        self.__result = None
        self.__whisper_cpp_path = whisper_cpp_path
        self.__backend = backend
        self.__logger = logger
        self.__speakers = speakers
        self.__diarization_pipeline = diarization_object
        self.__tokens_to_ignore = [
            "<|nospeech|>",
            "<|p>",
            "<|>",
            '"',
        ]

        if backend == "hf":
            self.__hf_init()

    def __hf_init(self) -> None:
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.__model_name,
            torch_dtype=self.__torch_dtype,
            use_safetensors=True,
            cache_dir="cache",
        )
        self.model.to(self.__device)
        self.processor = AutoProcessor.from_pretrained(self.__model_name)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.__model_name,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.__torch_dtype,
            device=self.__device,
            return_timestamps=True,
        )

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
        seconds = float(seconds)  # ensure it's a float
        millis = int(round((seconds % 1) * 1000))
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def __run_cmd(
        self, command: list, input_data: Optional[bytes] = None, pass_fds: tuple = ()
    ) -> Optional[bytes]:
        """
        Run a command using subprocess.run.
        Optionally pipes input_data to stdin.
        Returns stdout bytes on success, None on failure.
        """
        try:
            command_str = re.sub(r"/dev/fd/\d+", "<fd>", " ".join(command))
            self.__logger.debug(f"Running command: {command_str}")
            result = subprocess.run(
                command, input=input_data, capture_output=True, pass_fds=pass_fds
            )

            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=result.returncode,
                    cmd=command_str,
                    output=result.stdout.decode(),
                    stderr=result.stderr.decode(),
                )
        except Exception as e:
            self.__logger.error(f"Error running command: {e}")
            return None

        return result.stdout

    def __parse_timestamp(self, timestamp_str) -> Optional[float]:
        if timestamp_str is None:
            return None

        time_part, ms_part = timestamp_str.split(",")

        if not ms_part:
            ms_part = "0"

        # Split time part into hours, minutes, seconds
        hours, minutes, seconds = map(int, time_part.split(":"))

        # Convert to total seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds + int(ms_part) / 1000.0

        return total_seconds

    def __calculate_avg_score(self, tokens: list) -> Optional[float]:
        """
        Calculate the average probability score from token 'p' values.
        Excludes special tokens like [_BEG_].
        """
        scores = [
            token.get("p", 0)
            for token in tokens
            if token.get("text", "").strip()
            and not token.get("text", "").startswith("[")
        ]
        if scores:
            return round(sum(scores) / len(scores), 4)
        return None

    def __process_transcription(self, items, source: str) -> dict:
        """
        Normalize and process transcription items from either HF or
        whisper.cpp.
        """
        full_transcription = ""
        segments = []
        chunks = []

        for index, item in enumerate(items):
            text = item.get("text", "").strip()

            if not text:
                continue

            if text in self.__tokens_to_ignore:
                continue

            if source == "cpp":
                try:
                    text = bytes(text, "iso-8859-1").decode("utf-8")
                except UnicodeDecodeError:
                    self.__logger.error(
                        f"Failed to decode {text} from transcription, using ISO-8859-1 encoding."
                    )
                    continue

            if full_transcription and not full_transcription.endswith(" "):
                full_transcription += " "

            full_transcription += text

            # Calculate average score for cpp backend
            avg_score = None
            if source == "cpp" and "tokens" in item:
                avg_score = self.__calculate_avg_score(item["tokens"])

            if source == "hf":
                start, end = item["timestamp"]
                start_ms = self.__seconds_to_srt_time(str(start))
                end_ms = self.__seconds_to_srt_time(str(end))
                start_time = self.__parse_timestamp(start_ms)
                end_time = self.__parse_timestamp(end_ms)
                ts_ms = (start_ms, end_ms)

            else:
                if item["tokens"][0]["text"] == "[_BEG_]":
                    start_time_token = item["tokens"][1]["timestamps"]["from"]
                    start_time = self.__parse_timestamp(start_time_token)
                else:
                    start_time_token = item["tokens"][0]["timestamps"]["from"]
                    start_time = self.__parse_timestamp(start_time_token)

                end_time_token = item["tokens"][-1]["timestamps"]["to"]
                end_time = self.__parse_timestamp(end_time_token)

                if (end_time - start_time) < 1.5:
                    time_to_add = 1.5 - (end_time - start_time)
                    next_item_start_time = self.__parse_timestamp(
                        items[index + 1]["tokens"][0]["timestamps"]["from"]
                        if index + 1 < len(items)
                        else None
                    )

                    end_time += time_to_add

                    if next_item_start_time and end_time > next_item_start_time:
                        end_time = next_item_start_time - 0.1

                    end_time_token = self.__seconds_to_srt_time(str(end_time))

                ts_ms = (start_time_token, end_time_token)

            duration = end_time - start_time

            # If the text is longer than 90 characters, split it in the middle and adjust timestamps.
            if len(text) > 90:
                mid_index = len(text) // 2
                split_index = text.rfind(" ", 0, mid_index)

                if split_index == -1:
                    split_index = mid_index

                first_part = text[:split_index].strip()
                second_part = text[split_index:].strip()

                mid_time = start_time + duration / 2

                segments.append(
                    {
                        "start": start_time,
                        "end": mid_time,
                        "text": first_part,
                        "duration": mid_time - start_time,
                        "avg_score": avg_score,
                    }
                )

                segments.append(
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
                            self.__seconds_to_srt_time(str(mid_time)),
                        ),
                        "text": first_part,
                        "avg_score": avg_score,
                    }
                )

                chunks.append(
                    {
                        "timestamp": (mid_time, end_time),
                        "timestamp_ms": (
                            self.__seconds_to_srt_time(str(mid_time)),
                            ts_ms[1],
                        ),
                        "text": second_part,
                        "avg_score": avg_score,
                    }
                )

                continue

            segments.append(
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
            "segments": segments,
            "chunks": chunks,
            "speaker_count": 1,
        }

        self.__result = converted
        self.__transcribed_seconds = segments[-1]["end"] if segments else 0

        return converted

    def __transcribe_hf(self, filepath: Optional[str] = None) -> dict:
        if self.__audio_data:
            audio_array, sample_rate = self.__decode_wav_bytes(self.__audio_data)
            audio_input = {"raw": audio_array, "sampling_rate": sample_rate}
        else:
            audio_input = filepath

        result = self.pipe(
            audio_input,
            generate_kwargs={"task": "transcribe", "language": self.__language},
        )

        return self.__process_transcription(result.get("chunks", []), source="hf")

    def __transcribe_cpp(self, filepath: Optional[str] = None) -> dict:
        # -of - writes JSON to stdout (via /dev/stdout).
        # -np suppresses text output so stdout contains only JSON.
        # Use TemporaryFile + /dev/fd/N to avoid visible files on disk.
        temp_file = None
        try:
            if self.__audio_data:
                temp_file = tempfile.TemporaryFile()
                temp_file.write(self.__audio_data)
                temp_file.seek(0)
                fd = temp_file.fileno()
                wav_filepath = f"/dev/fd/{fd}"
                fds = (fd,)
            else:
                wav_filepath = filepath
                fds = ()

            command = [
                self.__whisper_cpp_path,
                "-l",
                self.__language,
                "-ojf",
                "-of",
                "-",
                "-np",
                "-m",
                self.__model_name,
                "-sns",
                "-fa",
                "-f",
                wav_filepath,
            ]

            json_str = self.__run_cmd(command, pass_fds=fds)
            if json_str is None:
                raise Exception("Failed to run whisper.cpp command")
        finally:
            if temp_file:
                temp_file.close()

        result = json.loads(json_str.decode("iso-8859-1"))
        return self.__process_transcription(
            result.get("transcription", []), source="cpp"
        )

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
            match self.__backend:
                case "hf":
                    self.__transcribe_hf(self.__audio_path)
                case "cpp":
                    self.__transcribe_cpp(self.__audio_path)
                case _:
                    raise ValueError(f"Unsupported backend: {self.__backend}")
        except Exception as e:
            self.__logger.error(f"Error during transcription: {str(e)}")
            return None

        if not self.__result:
            raise Exception("Transcription result is not available.")

        return self.__transcribed_seconds

    def diarization(self) -> dict:
        """
        Perform speaker diarization on the transcribed audio.
        """

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
            self.__logger.info("Initializing diarization pipeline...")
            self.__diarization_pipeline = diarization_init(self.__hf_token)
        else:
            self.__logger.info("Diarization pipeline already initialized.")

        if not self.__diarization_pipeline:
            self.__logger.error("Diarization pipeline initialization failed.")
            raise Exception("Diarization pipeline is not available.")

        if not self.__result:
            raise Exception(
                "Transcription result is not available. Please transcribe first."
            )

        self.__logger.info("Running diarization pipeline...")

        if self.__audio_data:
            audio_array, sample_rate = self.__decode_wav_bytes(self.__audio_data)
            waveform = torch.from_numpy(audio_array).unsqueeze(0)
            audio_input = {"waveform": waveform, "sample_rate": sample_rate}
        else:
            audio_input = self.__audio_path

        diarization = self.__diarization_pipeline(
            audio_input,
            num_speakers=int(self.__speakers),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        aligned_segments = self.__align_speakers(self.__result["chunks"], diarization)

        return {
            "full_transcription": self.__result["full_transcription"],
            "segments": aligned_segments,
            "speaker_count": len(list(diarization.speaker_diarization.labels()))
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
                "start": chunk_start,
                "end": chunk_end,
                "text": chunk_text.strip(),
                "speaker": dominant_speaker,
                "active_speakers": active_speakers,
                "duration": chunk_end - chunk_start,
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

        index = 0
        subtitles = ""

        for index, chunk in enumerate(self.__result["chunks"]):
            start, end = chunk["timestamp_ms"]
            text = chunk["text"].strip()

            if not text:
                continue

            caption = self.__caption_split(text)
            subtitles += f"{index + 1}\n"
            subtitles += f"{start} --> {end}\n"
            subtitles += f"{caption}\n\n"

        return subtitles

    def __format_timestamp(self, seconds) -> str:
        """
        Format a timestamp in seconds to MM:SS format.
        """
        hours = int(seconds // 3600)
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

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
        new_caption = f"{first_line}\n{second_line}"

        return new_caption


if __name__ == "__main__":
    logger = logging.getLogger("whisper_transcriber")

    audio_file = "test.wav"
    transcriber = WhisperAudioTranscriber(
        logger=logger,
        backend="cpp",
        audio_path=audio_file,
        model_name="models/sv_large.bin",
        language="sv",
        speakers=2,
    )

    transcribed_seconds = transcriber.transcribe()
    diarization_result = transcriber.diarization()

    print(diarization_result)
