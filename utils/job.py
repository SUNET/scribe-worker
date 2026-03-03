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
import requests
import subprocess
import wave

from enum import Enum
from pathlib import Path
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
        api_file_storage_dir: str,
        hf_whisper: Optional[bool] = False,
        hf_token: Optional[str] = None,
    ):
        self.logger = logger
        self.api_url = api_url
        self.api_file_storage_dir = api_file_storage_dir
        self.hf_whisper = hf_whisper
        self.hf_token = hf_token
        self.speakers = 0
        self.file_data = None

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
        self.file_data = None

        # Ensure the file storage directory exists
        Path(self.api_file_storage_dir).mkdir(parents=True, exist_ok=True)

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
        self.logger.info(f"  HF: {self.hf_whisper}")
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

        self.logger.debug("Transcoding file")
        if not self.__transcode_file():
            self.logger.error("Transcoding failed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="Transcoding failed",
                transcribed_seconds=None,
            )
            return False

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

        self.logger.debug("Downscaling file")
        if not self.__downscale_file():
            self.logger.error("Downscaling failed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="Downscaling failed",
                transcribed_seconds=None,
            )
            return False

        self.file_data = None

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
            "hf" if self.hf_whisper else "cpp",
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

    def __run_cmd_pipe(self, command: list, input_data: bytes) -> bool:
        """
        Run a command using subprocess.Popen, piping input_data to stdin.
        """
        try:
            command_str = " ".join(command)
            self.logger.debug(f"Executing piped command: {command_str}")
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
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

    def __downscale_file(self) -> bool:
        """
        Downscale videos to a smaller size.
        Output is captured in memory (self.mp4_data).
        """

        command = [
            settings.FFMPEG_PATH,
            "-i",
            "pipe:0",
            "-vf",
            "scale=-2:240:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-profile:v",
            "high",
            "-level",
            "3.1",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "48",
            "-keyint_min",
            "48",
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof",
            "-f",
            "mp4",
            "pipe:1",
        ]
        try:
            self.mp4_data = self.__run_cmd_pipe(command, self.file_data)
        except Exception as e:
            self.logger.error(f"Error during downscaling: {e}")
            return False

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

    def __transcode_file(self) -> bool:
        """
        Transcode the audio file using ffmpeg.
        The transcoded format should be 16kHz mono signed 16-bit PCM.
        Raw PCM is piped to stdout (WAV format requires seeking, which
        pipes don't support), then wrapped in a WAV header in memory.
        """

        command = [
            settings.FFMPEG_PATH,
            "-i",
            "pipe:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "s16le",
            "pipe:1",
        ]

        try:
            pcm_data = self.__run_cmd_pipe(command, self.file_data)
            self.wav_data = self.__raw_pcm_to_wav(pcm_data)
        except Exception as e:
            self.logger.error(f"Error during transcoding: {e}")
            return False

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
        Download the file from the API broker.
        """

        try:
            response = requests.get(
                f"{self.api_url}/{self.user_id}/{self.uuid}/file",
                cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
            )
            response.raise_for_status()

            if response.status_code != 200:
                self.logger.error(f"Error downloading file: {response.status_code}")
                raise Exception("File not downloaded")

            self.file_data = response.content

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

    def __upload_mp4(self, mp4_bytes: bytes) -> bool:
        """
        Upload the MP4 data to the API broker.
        """

        try:
            response = requests.put(
                f"{self.api_url}/{self.user_id}/{self.uuid}/file",
                files={"file": (f"{self.uuid}.mp4", io.BytesIO(mp4_bytes), "video/mp4")},
                cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
            )
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Error uploading MP4 file: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error uploading MP4 file: {e}")
            return False

    def __put_result(self) -> int:
        """
        Upload results to the API broker.
        All data is uploaded from memory.
        """
        header = {
            "Content-Type": "application/json",
        }

        results = {}
        if self.srt_data:
            results["srt"] = {"result": self.srt_data, "format": "srt"}
        if self.json_data:
            results["json"] = {"result": self.json_data, "format": "json"}

        for output_format, json_data in results.items():
            try:
                response = requests.put(
                    f"{self.api_url}/{self.user_id}/{self.uuid}/result",
                    json=json_data,
                    headers=header,
                    cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
                )
                response.raise_for_status()
            except requests.RequestException as e:
                self.logger.error(f"Error uploading {output_format}: {e}")
                return False

            self.logger.info(f"Uploaded {output_format} for job {self.uuid}")

        if self.mp4_data:
            try:
                self.__upload_mp4(self.mp4_data)
            except Exception as e:
                self.logger.error(f"Error uploading mp4: {e}")
                return False
            self.logger.info(f"Uploaded mp4 for job {self.uuid}")

        return True

    def __get_model(self) -> str:
        """
        Return the correct model file based on
        model type and language.
        """

        if self.hf_whisper:
            model = settings.WHISPER_MODELS_HF[self.language][self.model_type.lower()]
        else:
            model = (
                "models/"
                + settings.WHISPER_MODELS_CPP[self.language][self.model_type.lower()]
            )

        return model

    def __cleanup(self) -> bool:
        """
        Delete all files related to the job.
        """

        if not self.uuid:
            return

        self.file_data = None
        self.wav_data = None
        self.srt_data = None
        self.json_data = None
        self.mp4_data = None

        return True
