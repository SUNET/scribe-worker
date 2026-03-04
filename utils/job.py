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
import os
import re
import requests
import subprocess
import tempfile
import wave

from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Optional
from utils import settings
from utils.whisper import WhisperAudioTranscriber

settings = settings.get_settings()


class JobStatusEnum(str, Enum):
    """
    Enum representing the status of a job.
    """

    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptionJob:
    def __init__(
        self,
        logger: logging.Logger,
        api_url: str,
        hf_token: Optional[str] = None,
    ):
        self.logger = logger
        self.api_url = api_url
        self.hf_token = hf_token
        self.speakers = 0

    def __enter__(self) -> "TranscriptionJob":
        """
        Initialize the transcription job.
        """
        self.uuid = None
        self.user_id = None
        self.language = None
        self.model_type = None
        self.model = None
        self.filename = None
        self.speakers = 0
        self.mp4_data = None
        self.mp3_data = None
        self.__temp_file = None

        return self

    def __exit__(self, *args: object) -> None:
        """
        Cleanup resources when the job is done.
        """
        self.__cleanup()

    def start(self) -> bool:
        """
        Start the transcription job.
        """

        job = self.__get_job()
        if not job:
            return

        self.uuid = job.get("uuid")
        self.user_id = job.get("user_id")
        self.language = job.get("language")
        self.model_type = job.get("model_type")
        self.model = self.__get_model()
        self.speakers = job.get("speakers", 0)
        self.filename = self.uuid
        self.output_format = job.get("output_format", "txt")

        if not self.speakers:
            self.speakers = 0

        self.logger.info(f"Starting transcription job {self.uuid}")
        self.logger.info(f"  User: {self.user_id}")
        self.logger.info(f"  Language: {self.language}")
        self.logger.info(f"  Model: {self.model}")
        self.logger.info(f"  Model type: {self.model_type}")
        self.logger.info(f"  Filename: {self.filename}")
        self.logger.info(f"  Speakers: {self.speakers}")
        self.logger.info(f"  Output format: {self.output_format}")

        self.logger.debug("Updating job status to IN_PROGRESS")
        self.__put_status(
            JobStatusEnum.IN_PROGRESS, error=None, transcribed_seconds=None
        )

        self.logger.debug("Fetching file from API broker")
        if not self.__get_file():
            self.logger.error("File download failed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="File download failed",
                transcribed_seconds=None,
            )
            return False

        has_video = self.__has_video_stream()

        # Run transcode + downscale/downsample in parallel
        self.logger.debug("Transcoding and preparing preview in parallel")
        with ThreadPoolExecutor(max_workers=2) as pool:
            transcode_future = pool.submit(self.__transcode_file)
            if has_video:
                preview_future = pool.submit(self.__downscale_video)
                preview_label = "Downscaling"
            else:
                preview_future = pool.submit(self.__downsample_audio)
                preview_label = "Audio downsampling"

            if not transcode_future.result():
                self.logger.error("Transcoding failed")
                self.__put_status(
                    JobStatusEnum.FAILED,
                    error="Transcoding failed",
                    transcribed_seconds=None,
                )
                return False

            if not preview_future.result():
                self.logger.error(f"{preview_label} failed")
                self.__put_status(
                    JobStatusEnum.FAILED,
                    error=f"{preview_label} failed",
                    transcribed_seconds=None,
                )
                return False

        # Close the temp file (auto-deleted) before transcription.
        self.__close_temp_file()

        self.logger.debug("Transcribing file")
        transcribed_seconds = self.__transcribe()

        if not transcribed_seconds:
            self.logger.error("Transcription failed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="Transcription failed",
                transcribed_seconds=None,
            )
            return False

        self.logger.debug("Uploading results to backend")
        if not self.__put_result():
            self.logger.error("File upload failed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="File upload failed",
                transcribed_seconds=None,
            )
            return False

        self.logger.info(f"Job {self.uuid} completed successfully")
        self.__put_status(
            JobStatusEnum.COMPLETED, error=None, transcribed_seconds=transcribed_seconds
        )
        self.logger.info(
            f"Transcription completed, total transcribed seconds: {transcribed_seconds}"
        )

        return True

    def __transcribe(self) -> bool:
        """
        Transcribe the audio file using Hugging Face Whisper.
        """
        self.logger.info("Starting transcription")
        transcriber = WhisperAudioTranscriber(
            self.logger,
            audio_data=self.wav_data,
            model_name=self.model,
            language=self.language,
            speakers=self.speakers,
            hf_token=self.hf_token,
        )

        transcribed_seconds = transcriber.transcribe()

        if transcribed_seconds is None:
            return None

        self.srt_data = transcriber.subtitles()

        if self.output_format == "txt":
            drz = transcriber.diarization()
            self.json_data = dict(drz) if drz else None
        else:
            self.json_data = None

        self.wav_data = None

        return transcribed_seconds

    def __run_cmd(self, command: list) -> bool:
        """
        Run a command using subprocess.run.
        Raises an exception if the command fails.
        """
        try:
            command_str = " ".join(command)
            self.logger.debug(f"Executing command: {command_str}")
            result = subprocess.run(command, capture_output=True)

            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=result.returncode,
                    cmd=command_str,
                    output=result.stdout.decode(),
                    stderr=result.stderr.decode(),
                )
        except Exception as e:
            self.logger.error(f"Error when executing command: {e}")
            raise e

        return True

    def __run_cmd_pipe(self, command: list, input_data: bytes = None, pass_fds: tuple = ()) -> bytes:
        """
        Run a command, capturing stdout.
        Optionally pipes input_data to stdin.
        Optionally passes file descriptors to the child process.
        """
        try:
            command_str = re.sub(r"/dev/fd/\d+", "<fd>", " ".join(command))
            self.logger.debug(f"Executing piped command: {command_str}")
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=pass_fds,
            )
            stdout, stderr = process.communicate(input=input_data)

            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=process.returncode,
                    cmd=command_str,
                    output=None,
                    stderr=stderr.decode(),
                )
        except Exception as e:
            self.logger.error(f"Error when executing piped command: {e}")
            raise e

        return stdout

    def __downscale_video(self) -> bool:
        """
        Downscale videos to a smaller size.
        Output is captured in memory (self.mp4_data).
        """
        input_path, fd = self.__ffmpeg_input_fd()
        command = [
            settings.FFMPEG_PATH,
            "-nostdin",
            "-threads", "0",
            "-i",
            input_path,
            "-vf",
            "scale=-2:240:flags=bilinear",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-profile:v",
            "baseline",
            "-pix_fmt",
            "yuv420p",
            "-g", "24",
            "-keyint_min", "24",
            "-c:a",
            "aac",
            "-b:a",
            "48k",
            "-ar",
            "44100",
            "-ac",
            "1",
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof",
            "-f",
            "mp4",
            "pipe:1",
        ]
        try:
            self.mp4_data = self.__run_cmd_pipe(command, pass_fds=(fd,))
        except Exception as e:
            self.logger.error(f"Error during downscaling: {e}")
            return False
        finally:
            os.close(fd)

        return True

    def __downsample_audio(self) -> bool:
        """
        Downsample audio to a lightweight MP3 preview.
        Output is captured in memory (self.mp3_data).
        """
        input_path, fd = self.__ffmpeg_input_fd()
        command = [
            settings.FFMPEG_PATH,
            "-nostdin",
            "-threads", "0",
            "-i",
            input_path,
            "-c:a",
            "libmp3lame",
            "-b:a",
            "64k",
            "-ar",
            "44100",
            "-ac",
            "1",
            "-f",
            "mp3",
            "pipe:1",
        ]
        try:
            self.mp3_data = self.__run_cmd_pipe(command, pass_fds=(fd,))
            self.mp4_data = None
        except Exception as e:
            self.logger.error(f"Error during audio downsampling: {e}")
            return False
        finally:
            os.close(fd)

        return True

    def __raw_pcm_to_wav(self, pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
        """
        Wrap raw PCM (s16le) data in a WAV header.
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    def __has_video_stream(self) -> bool:
        """
        Probe the temp file with ffprobe to check for a video stream.
        """
        input_path, fd = self.__ffmpeg_input_fd()
        command = [
            "ffprobe",
            "-v", "quiet",
            "-select_streams", "v",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            input_path,
        ]
        try:
            output = self.__run_cmd_pipe(command, pass_fds=(fd,))
            return b"video" in output
        except Exception:
            return False
        finally:
            os.close(fd)

    def __ffmpeg_input_fd(self) -> tuple:
        """
        Create a duplicate fd (independently seekable) and return (path, fd).
        Caller must close the fd after use.
        """
        fd = os.dup(self.__temp_file.fileno())
        os.lseek(fd, 0, os.SEEK_SET)
        return f"/dev/fd/{fd}", fd

    def __close_temp_file(self):
        if self.__temp_file:
            self.__temp_file.close()
            self.__temp_file = None

    def __transcode_file(self) -> bool:
        """
        Transcode the audio file using ffmpeg.
        The transcoded format should be 16kHz mono signed 16-bit PCM.
        Raw PCM is piped to stdout, then wrapped in a WAV header in memory.
        """
        input_path, fd = self.__ffmpeg_input_fd()
        command = [
            settings.FFMPEG_PATH,
            "-nostdin",
            "-threads", "0",
            "-i",
            input_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "s16le",
            "pipe:1",
        ]

        try:
            pcm_data = self.__run_cmd_pipe(command, pass_fds=(fd,))
            self.wav_data = self.__raw_pcm_to_wav(pcm_data)
        except Exception as e:
            self.logger.error(f"Error during transcoding: {e}")
            return False
        finally:
            os.close(fd)

        return True

    def __get_job(self) -> dict:
        """
        Get the next job from the API broker.
        """
        try:
            response = requests.get(
                f"{self.api_url}/next",
                cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
            )
            response.raise_for_status()
            job = response.json()["result"]
            if "status" in job and job["status"] != JobStatusEnum.IN_PROGRESS:
                self.logger.info(f"Job {job['uuid']} is not in_progress. Skipping.")
                return {}

        except Exception as e:
            self.logger.error(f"Error fetching next job: {e}")
            return {}

        return job

    def __get_file(self) -> bool:
        """
        Download the file from the API broker, streaming directly to temp file.
        """

        try:
            response = requests.get(
                f"{self.api_url}/{self.user_id}/{self.uuid}/file",
                cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
                stream=True,
            )
            response.raise_for_status()

            self.__temp_file = tempfile.TemporaryFile()
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                self.__temp_file.write(chunk)

        except Exception as e:
            self.logger.error(f"Error downloading file: {e}")
            return False

        return True

    def __put_status(
        self, status: JobStatusEnum, error: str, transcribed_seconds: int
    ) -> bool:
        """
        Update the job status in the API broker.
        """

        try:
            response = requests.put(
                f"{self.api_url}/{self.uuid}",
                json={
                    "status": status,
                    "error": error,
                    "transcribed_seconds": transcribed_seconds,
                },
                cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
            )
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Error updating job status: {e}")
            return False

        return True

    def __upload_media(self) -> None:
        """
        Upload the media preview (MP4 or MP3) to the API broker.
        """
        if self.mp4_data:
            filename = f"{self.uuid}.mp4"
            data = self.mp4_data
            mime = "video/mp4"
        elif self.mp3_data:
            filename = f"{self.uuid}.mp3"
            data = self.mp3_data
            mime = "audio/mpeg"
        else:
            return

        response = requests.put(
            f"{self.api_url}/{self.user_id}/{self.uuid}/file",
            files={"file": (filename, io.BytesIO(data), mime)},
            cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
        )
        response.raise_for_status()
        self.mp4_data = None
        self.mp3_data = None

    def __upload_result(self, output_format: str, json_data: dict) -> None:
        """
        Upload a single result (srt/json) to the API broker.
        """
        response = requests.put(
            f"{self.api_url}/{self.user_id}/{self.uuid}/result",
            json=json_data,
            headers={"Content-Type": "application/json"},
            cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
        )
        response.raise_for_status()

    def __put_result(self) -> bool:
        """
        Upload all results to the API broker in parallel.
        """
        futures = {}

        with ThreadPoolExecutor(max_workers=3) as pool:
            if self.srt_data:
                futures[pool.submit(
                    self.__upload_result, "srt", {"result": self.srt_data, "format": "srt"}
                )] = "srt"
            if self.json_data:
                futures[pool.submit(
                    self.__upload_result, "json", {"result": self.json_data, "format": "json"}
                )] = "json"
            if self.mp4_data or self.mp3_data:
                futures[pool.submit(self.__upload_media)] = "media"

            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                    self.logger.info(f"Uploaded {name} for job {self.uuid}")
                except Exception as e:
                    self.logger.error(f"Error uploading {name}: {e}")
                    return False

        return True

    def __get_model(self) -> str:
        """
        Return the correct model name based on
        model type and language.
        """
        return settings.WHISPER_MODELS_HF[self.language][self.model_type.lower()]

    def __cleanup(self) -> bool:
        """
        Delete all files related to the job.
        """

        if not self.uuid:
            return

        self.wav_data = None
        self.srt_data = None
        self.json_data = None
        self.mp4_data = None
        self.mp3_data = None
        self.__close_temp_file()

        return True
