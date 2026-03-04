import io
import logging
import numpy as np
import os
import torch
import warnings
import wave
import whisper_timestamped as whisper

warnings.filterwarnings("ignore", module="pyannote")

from huggingface_hub import snapshot_download
from pyannote.audio import Pipeline
from typing import Optional
from utils.settings import get_settings

settings = get_settings()

if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN

os.environ["TORCH_HUB_TRUST_REPO"] = "1"
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"


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


_model_cache: dict[str, object] = {}


def load_whisper_model(model_name: str, logger: logging.Logger) -> object:
    """
    Load and cache a whisper model. Returns cached model if already loaded.
    """
    if model_name in _model_cache:
        logger.info(f"Using cached model '{model_name}'")
        return _model_cache[model_name]

    device, _ = get_torch_device()

    # Parse optional revision (e.g. "kblab/kb-whisper-large@strict")
    revision = None
    load_name = model_name
    if "@" in load_name:
        load_name, revision = load_name.rsplit("@", 1)

    # If a specific revision is needed, download via huggingface_hub first
    # and pass the local path to whisper-timestamped (bypasses its revision=None)
    if revision and "/" in load_name:
        load_name = snapshot_download(load_name, revision=revision)
        logger.info(f"Using '{model_name}' revision '{revision}' from {load_name}")

    try:
        model = whisper.load_model(load_name, device=device)
    except NotImplementedError:
        logger.warning(f"Failed to load model on {device}, falling back to CPU")
        device = "cpu"
        model = whisper.load_model(load_name, device=device)

    logger.info(f"Loaded model '{model_name}' on {device}")
    _model_cache[model_name] = model
    return model


class WhisperAudioTranscriber:
    def __init__(
        self,
        logger: logging.Logger,
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
        self.__device, self.__torch_dtype = get_torch_device()
        self.__language = language.split("(")[0].strip().lower() if language else language
        self.__result = None
        self.__logger = logger
        self.__speakers = speakers
        self.__diarization_pipeline = diarization_object
        self.__model_name = model_name
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
                confidences = [float(w.get("confidence", 0)) for w in words if w.get("text", "").strip()]
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
        self.__transcribed_seconds = processed_segments[-1]["end"] if processed_segments else 0

        return converted

    def __transcribe_audio(self, filepath: Optional[str] = None) -> dict:
        if self.__audio_data:
            audio, _ = self.__decode_wav_bytes(self.__audio_data)
        else:
            audio = whisper.load_audio(filepath)

        result = whisper.transcribe(
            self.__model,
            audio,
            language=self.__language,
            vad=True,
            beam_size=1,
            condition_on_previous_text=False,
            fp16=self.__device != "cpu",
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
        Format a timestamp in seconds to HH:MM:SS format.
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
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("whisper_transcriber")

    audio_file = "test.wav"
    transcriber = WhisperAudioTranscriber(
        logger=logger,
        audio_path=audio_file,
        model_name="base",
        language="sv",
        speakers=2,
    )

    transcribed_seconds = transcriber.transcribe()
    print(f"Transcribed seconds: {transcribed_seconds}")

    subtitles = transcriber.subtitles()
    print(f"Subtitles:\n{subtitles}")
