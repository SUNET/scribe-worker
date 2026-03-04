import io
import logging
import requests
import subprocess
import tempfile
import time
import wave

from concurrent.futures import ThreadPoolExecutor
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
        Transcribe the audio file using Hugging Face Whisper.
        """
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
            self.json_data = drz if drz else None
        else:
            self.json_data = None

        self.wav_data = None

        return transcribed_seconds

    def __run_cmd_pipe(self, command: list, input_data: bytes = None) -> bytes:
        """
        Run a command, capturing stdout.
        Optionally pipes input_data to stdin.
        """
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if input_data else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(input=input_data)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode,
                cmd=" ".join(command),
                output=None,
                stderr=stderr.decode(),
            )

        return stdout

    def __downscale_video(self) -> bool:
        """
        Downscale videos to a smaller size.
        Output is captured in memory (self.mp4_data).
        """
        input_path = self.__ffmpeg_input_path()
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
            self.mp4_data = self.__run_cmd_pipe(command)
        except Exception as e:
            self.logger.error(f"Error during downscaling: {e}")
            return False
        return True

    def __downsample_audio(self) -> bool:
        """
        Downsample audio to a lightweight AAC stream in a fragmented MP4 container.
        This ensures the same streaming support as video previews.
        Output is captured in memory (self.mp4_data).
        """
        input_path = self.__ffmpeg_input_path()
        command = [
            settings.FFMPEG_PATH,
            "-nostdin",
            "-threads", "0",
            "-i",
            input_path,
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
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
            self.mp4_data = self.__run_cmd_pipe(command)
        except Exception as e:
            self.logger.error(f"Error during audio downsampling: {e}")
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

    def __has_video_stream(self) -> bool:
        """
        Probe the temp file with ffprobe to check for a video stream.
        """
        input_path = self.__ffmpeg_input_path()
        command = [
            "ffprobe",
            "-v", "quiet",
            "-select_streams", "v",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            input_path,
        ]
        try:
            output = self.__run_cmd_pipe(command)
            return b"video" in output
        except Exception:
            return False

    def __ffmpeg_input_path(self) -> str:
        """
        Return the path to the temp file for ffmpeg input.
        """
        return self.__temp_file.name

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
        input_path = self.__ffmpeg_input_path()
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
            pcm_data = self.__run_cmd_pipe(command)
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
            files={"file": (f"{self.uuid}.mp4", io.BytesIO(self.mp4_data), "video/mp4")},
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
                self.__upload_result("json", {"result": self.json_data, "format": "json"})

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
