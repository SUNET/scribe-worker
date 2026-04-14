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
import multiprocessing as mp
import os
import requests
import tempfile
import time


from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Optional
from utils import settings

from utils.media import (
    downsample_audio,
    downscale_video,
    has_video_stream,
    transcode_to_wav,
)

from utils.whisper import WhisperAudioTranscriber
from utils.log import get_logger

log = get_logger()
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


def _transcribe_worker(
    wav_data, model, language, speakers, hf_token, output_format, result_dict
):
    """
    Run transcription in a child process so all memory (RAM + VRAM)
    is reclaimed by the OS when the process exits.
    """

    transcriber = WhisperAudioTranscriber(
        audio_data=wav_data,
        model_name=model,
        language=language,
        speakers=speakers,
        hf_token=hf_token,
    )

    transcribed_seconds = transcriber.transcribe()

    if transcribed_seconds is None:
        result_dict["transcribed_seconds"] = None
        return

    result_dict["transcribed_seconds"] = transcribed_seconds
    result_dict["srt_data"] = transcriber.subtitles()

    if output_format == "txt":
        drz = transcriber.diarization()
        result_dict["json_data"] = drz if drz else None
    else:
        result_dict["json_data"] = None


class TranscriptionJob:
    def __init__(
        self,
        logger: logging.Logger,
        api_url: str,
        hf_token: Optional[str] = None,
        active_jobs: Optional[dict] = None,
    ):
        self.logger = logger
        self.api_url = api_url
        self.hf_token = hf_token
        self._active_jobs = active_jobs

    def __enter__(self) -> "TranscriptionJob":
        """
        Initialize the transcription job.
        """
        self.uuid = None
        self.user_id = None
        self.language = None
        self.model_type = None
        self.model = None
        self.speakers = 0
        self.mp4_data = None
        self.__temp_file = None

        return self

    def __exit__(self, *args: object) -> None:
        """
        Cleanup resources when the job is done.
        """
        if self._active_jobs is not None:
            self._active_jobs.pop(os.getpid(), None)
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
        self.output_format = job.get("output_format", "txt")

        if self._active_jobs is not None and self.uuid:
            self._active_jobs[os.getpid()] = self.uuid

        if not self.speakers:
            self.speakers = 0

        self.logger.info(
            f"Job {self.uuid}: language={self.language}, model={self.model}, "
            f"speakers={self.speakers}, format={self.output_format}"
        )

        self.__put_status(
            JobStatusEnum.IN_PROGRESS, error=None, transcribed_seconds=None
        )

        try:
            return self.__process()
        except Exception:
            self.logger.exception(f"Job {self.uuid}: crashed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="Internal error",
                transcribed_seconds=None,
            )
            return False

    def __timed(self, label: str, func, *args, **kwargs):
        """Run func and log elapsed time at debug level."""
        t0 = time.monotonic()
        result = func(*args, **kwargs)
        elapsed = time.monotonic() - t0
        self.logger.debug(f"Job {self.uuid}: {label} took {elapsed:.2f}s")
        return result

    def __process(self) -> bool:
        """
        Run the job pipeline: download, transcode, transcribe, upload.
        """
        t0 = time.monotonic()

        if not self.__timed("download", self.__get_file):
            self.logger.error(f"Job {self.uuid}: file download failed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="File download failed",
                transcribed_seconds=None,
            )
            return False

        if not self.__timed("transcode", self.__transcode_file):
            self.logger.error(f"Job {self.uuid}: transcoding failed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="Transcoding failed",
                transcribed_seconds=None,
            )
            return False

        if self.__has_video_stream():
            if not self.__timed("downscale", self.__downscale_video):
                self.logger.error(f"Job {self.uuid}: downscaling failed")
                self.__put_status(
                    JobStatusEnum.FAILED,
                    error="Downscaling failed",
                    transcribed_seconds=None,
                )
                return False
        else:
            if not self.__timed("downsample", self.__downsample_audio):
                self.logger.error(f"Job {self.uuid}: audio downsampling failed")
                self.__put_status(
                    JobStatusEnum.FAILED,
                    error="Audio downsampling failed",
                    transcribed_seconds=None,
                )
                return False

        # Close the temp file (auto-deleted) before transcription.
        self.__close_temp_file()

        transcribed_seconds = self.__timed("transcribe", self.__transcribe)

        if not transcribed_seconds:
            self.logger.error(f"Job {self.uuid}: transcription failed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="Transcription failed",
                transcribed_seconds=None,
            )
            return False

        if not self.__timed("upload", self.__put_result):
            self.logger.error(f"Job {self.uuid}: upload failed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="File upload failed",
                transcribed_seconds=None,
            )
            return False

        self.__put_status(
            JobStatusEnum.COMPLETED, error=None, transcribed_seconds=transcribed_seconds
        )
        elapsed = time.monotonic() - t0
        self.logger.info(
            f"Job {self.uuid}: completed, {transcribed_seconds:.0f}s transcribed in {elapsed:.1f}s"
        )

        return True

    def __transcribe(self) -> bool:
        """
        Transcribe the audio file in a subprocess to avoid memory leaks.
        """
        result_dict = mp.Manager().dict()

        p = mp.Process(
            target=_transcribe_worker,
            args=(
                self.wav_data,
                self.model,
                self.language,
                self.speakers,
                self.hf_token,
                self.output_format,
                result_dict,
            ),
        )
        p.start()
        p.join()

        self.wav_data = None

        if p.exitcode != 0:
            self.logger.error(
                f"Job {self.uuid}: transcription subprocess exited with code {p.exitcode}"
            )
            return None

        transcribed_seconds = result_dict.get("transcribed_seconds")
        if transcribed_seconds is None:
            return None

        self.srt_data = result_dict.get("srt_data")
        self.json_data = result_dict.get("json_data")

        return transcribed_seconds

    def __downscale_video(self) -> bool:
        try:
            self.mp4_data = downscale_video(self.__temp_file.name)
        except Exception as e:
            self.logger.error(f"Error during downscaling: {e}")
            return False
        return True

    def __has_video_stream(self) -> bool:
        return has_video_stream(self.__temp_file.name)

    def __downsample_audio(self) -> bool:
        try:
            self.mp4_data = downsample_audio(self.__temp_file.name)
        except Exception as e:
            self.logger.error(f"Error during audio downsampling: {e}")
            self.mp4_data = None
            return False

        if not self.mp4_data:
            self.logger.error("Audio downsampling produced no output")
            self.mp4_data = None
            return False

        return True

    def __close_temp_file(self):
        if self.__temp_file:
            self.__temp_file.close()
            self.__temp_file = None

    def __transcode_file(self) -> bool:
        try:
            self.wav_data = transcode_to_wav(self.__temp_file.name)
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
                return {}

        except Exception:
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

            self.__temp_file = tempfile.NamedTemporaryFile(delete=True)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                self.__temp_file.write(chunk)
            self.__temp_file.flush()

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
        Upload the media preview (MP4) to the API broker.
        """
        if not self.mp4_data:
            return

        response = requests.put(
            f"{self.api_url}/{self.user_id}/{self.uuid}/file",
            files={
                "file": (f"{self.uuid}.mp4", io.BytesIO(self.mp4_data), "video/mp4")
            },
            cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
        )
        response.raise_for_status()
        self.mp4_data = None

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
        if response.status_code != 200:
            self.logger.error(f"Upload {output_format} response: {response.text}")
        response.raise_for_status()

    def __put_result(self) -> bool:
        """
        Upload results to the API broker.
        SRT and JSON are uploaded sequentially (same endpoint),
        media upload runs in parallel with the results.
        """
        media_future = None
        if self.mp4_data:
            pool = ThreadPoolExecutor(max_workers=1)
            media_future = pool.submit(self.__upload_media)

        try:
            if self.srt_data:
                self.__upload_result("srt", {"result": self.srt_data, "format": "srt"})

            if self.json_data:
                self.__upload_result(
                    "json", {"result": self.json_data, "format": "json"}
                )

            if media_future:
                media_future.result()
        except Exception as e:
            self.logger.error(f"Error uploading results: {e}")
            return False
        finally:
            if media_future:
                pool.shutdown(wait=False)

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
        self.__close_temp_file()

        return True
