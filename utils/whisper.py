import io
import logging
import numpy as np
import os
import re
import time
import torch
import warnings
import wave

from huggingface_hub import snapshot_download
from pyannote.audio import Pipeline
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from typing import Optional
from utils.settings import get_settings

warnings.filterwarnings("ignore", module="pyannote")
warnings.filterwarnings("ignore", module="transformers")
settings = get_settings()

if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN

os.environ["TORCH_HUB_TRUST_REPO"] = "1"
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


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


_model_cache: dict[str, object] = {}
_diarization_cache: Optional[Pipeline] = None
_vad_model = None


def diarization_init(hf_token: str) -> Optional[Pipeline]:
    """
    Initializes the diarization pipeline using HuggingFace's PyAnnote.
    Uses the community version for better performance.
    Returns cached pipeline if already loaded.
    """
    global _diarization_cache
    if _diarization_cache is not None:
        return _diarization_cache

    device, _ = get_torch_device()

    _diarization_cache = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=hf_token
    ).to(torch.device(device))

    return _diarization_cache


def _load_silero_vad():
    """Load and cache the Silero VAD model."""
    global _vad_model
    if _vad_model is not None:
        return _vad_model

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    _vad_model = (model, utils)
    return _vad_model


def _run_vad(audio: np.ndarray, sample_rate: int = 16000) -> list[tuple[float, float]]:
    """
    Run Silero VAD on audio and return speech segment boundaries as
    list of (start_seconds, end_seconds).

    Merges segments with gaps < 0.3s and pads each by 0.1s.
    Falls back to treating entire audio as one segment on failure.
    """
    audio_duration = len(audio) / sample_rate

    # Skip VAD for very short audio
    if audio_duration < 1.0:
        return [(0.0, audio_duration)]

    try:
        model, utils = _load_silero_vad()
        get_speech_timestamps = utils[0]

        audio_tensor = torch.from_numpy(audio).float()

        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=sample_rate,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )

        if not speech_timestamps:
            return []

        # Convert sample indices to seconds
        segments = [
            (ts["start"] / sample_rate, ts["end"] / sample_rate)
            for ts in speech_timestamps
        ]

        # Merge segments with gaps < 0.3s, but cap at 28s to stay within
        # Whisper's 30s window
        MAX_SEGMENT_DURATION = 28.0
        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end < 0.3 and (end - prev_start) <= MAX_SEGMENT_DURATION:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        # Pad each segment by 0.1s on each side
        padded = []
        for start, end in merged:
            padded.append((max(0.0, start - 0.1), min(audio_duration, end + 0.1)))

        return padded

    except Exception:
        return [(0.0, audio_duration)]


def load_whisper_model(model_name: str, logger: logging.Logger) -> tuple:
    """
    Load and cache a Whisper model using HF Transformers directly.
    Returns (model, processor) tuple.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    device, torch_dtype = get_torch_device()

    # Parse optional revision (e.g. "kblab/kb-whisper-large@strict")
    revision = None
    load_name = model_name
    if "@" in load_name:
        load_name, revision = load_name.rsplit("@", 1)

    is_hf_model = "/" in load_name

    # For non-HF models (e.g. "base", "large-v3"), map to openai repo
    if not is_hf_model:
        load_name = f"openai/whisper-{load_name}"

    # If a specific revision is needed, download via huggingface_hub first
    if revision:
        load_name = snapshot_download(
            load_name,
            revision=revision,
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin"],
        )
        logger.info(f"Using '{model_name}' revision '{revision}' from {load_name}")

    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            load_name, torch_dtype=torch_dtype
        ).to(device)
        processor = WhisperProcessor.from_pretrained(load_name)
    except (NotImplementedError, RuntimeError):
        logger.warning(f"Failed to load model on {device}, falling back to CPU")
        device = "cpu"
        torch_dtype = torch.float32
        model = WhisperForConditionalGeneration.from_pretrained(
            load_name, torch_dtype=torch_dtype
        ).to(device)
        processor = WhisperProcessor.from_pretrained(load_name)

    logger.info(f"Loaded model '{model_name}' on {device}")
    result = (model, processor)
    _model_cache[model_name] = result
    return result


def _tokens_to_words(processor, token_ids, token_timestamps) -> list[dict]:
    """
    Convert token-level IDs and timestamps from model.generate() (DTW mode)
    into word-level entries: [{"text": ..., "start": ..., "end": ...}, ...]
    """
    words = []
    current_tokens = []  # list of (decoded_text, timestamp)

    special_ids = set(processor.tokenizer.all_special_ids)

    for tid, ts in zip(token_ids, token_timestamps):
        tid = int(tid)
        ts = float(ts)

        if tid in special_ids:
            continue

        decoded = processor.tokenizer.decode([tid])

        # Space prefix marks a word boundary in GPT-2/Whisper tokenizers
        if decoded.startswith(" ") and current_tokens:
            word_text = "".join(t for t, _ in current_tokens).strip()
            if word_text:
                words.append(
                    {"text": word_text, "start": current_tokens[0][1], "end": ts}
                )
            current_tokens = [(decoded, ts)]
        else:
            current_tokens.append((decoded, ts))

    if current_tokens:
        word_text = "".join(t for t, _ in current_tokens).strip()
        if word_text:
            words.append(
                {
                    "text": word_text,
                    "start": current_tokens[0][1],
                    "end": current_tokens[-1][1],
                }
            )

    return words


def _parse_timestamp_tokens(processor, token_ids) -> list[dict]:
    """
    Parse Whisper's timestamp tokens from generate() output into segments.
    Fallback when DTW token timestamps are not available.

    Whisper encodes timestamps as special tokens: <|0.00|>, <|0.02|>, ...
    Token ID = timestamp_begin + (seconds / 0.02).
    """
    tokenizer = processor.tokenizer
    timestamp_begin = tokenizer.convert_tokens_to_ids("<|0.00|>")

    segments = []
    current_text_tokens = []
    current_start = None

    for tid in token_ids:
        tid = int(tid)

        # Skip non-timestamp special tokens (lang, task, BOS, EOS, etc.)
        if tid in tokenizer.all_special_ids and tid < timestamp_begin:
            continue

        if tid >= timestamp_begin:
            timestamp = round((tid - timestamp_begin) * 0.02, 2)
            if current_start is None:
                current_start = timestamp
            elif current_text_tokens:
                text = tokenizer.decode(current_text_tokens, skip_special_tokens=True).strip()
                if text:
                    segments.append(
                        {"start": current_start, "end": timestamp, "text": text}
                    )
                current_text_tokens = []
                current_start = timestamp
        else:
            current_text_tokens.append(tid)

    return segments


def _group_words_into_segments(words: list[dict]) -> list[dict]:
    """
    Group word-level timestamps into subtitle-sized segments.

    Breaks on:
    - Pause > 0.8s between words
    - Sentence-ending punctuation (.?!) when accumulated text > 40 chars
    - Hard break at ~90 chars
    """
    if not words:
        return []

    segments = []
    current_words = []
    current_text = ""
    seg_start = None

    for word in words:
        w_text = word.get("text", "").strip()
        w_start = word.get("start", 0.0)
        w_end = word.get("end", 0.0)

        if not w_text:
            continue

        # Check if we should break before adding this word
        if current_words:
            prev_end = current_words[-1]["end"]
            gap = w_start - prev_end

            should_break = False

            # Break on pause > 0.8s
            if gap > 0.8:
                should_break = True

            # Break on sentence-ending punctuation when text is long enough
            if (
                len(current_text) > 40
                and current_text
                and current_text[-1] in ".?!"
            ):
                should_break = True

            # Hard break at ~90 chars
            if len(current_text) + 1 + len(w_text) > 90:
                should_break = True

            if should_break:
                segments.append(
                    {
                        "start": seg_start,
                        "end": current_words[-1]["end"],
                        "text": current_text.strip(),
                        "words": current_words,
                    }
                )
                current_words = []
                current_text = ""
                seg_start = None

        if seg_start is None:
            seg_start = w_start

        current_words.append({"text": w_text, "start": w_start, "end": w_end})
        if current_text:
            current_text += " "
        current_text += w_text

    # Flush remaining words
    if current_words:
        segments.append(
            {
                "start": seg_start,
                "end": current_words[-1]["end"],
                "text": current_text.strip(),
                "words": current_words,
            }
        )

    return segments


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
        self.__device, _ = get_torch_device()
        self.__language = (
            language.split("(")[0].strip().lower() if language else language
        )
        self.__result = None
        self.__logger = logger
        self.__speakers = speakers
        self.__diarization_pipeline = diarization_object
        self.__model, self.__processor = load_whisper_model(model_name, logger)

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
        Normalize and process transcription segments.
        """
        full_transcription = ""
        processed_segments = []
        chunks = []

        for segment in segments:
            text = re.sub(r" {2,}", " ", segment.get("text", "")).strip()

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
                    }
                )

                processed_segments.append(
                    {
                        "start": mid_time,
                        "end": end_time,
                        "text": second_part,
                        "duration": end_time - mid_time,
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
                    }
                )

                continue

            processed_segments.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "duration": duration,
                }
            )

            chunks.append(
                {
                    "timestamp": (start_time, end_time),
                    "timestamp_ms": ts_ms,
                    "text": text,
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
            audio, sample_rate = self.__decode_wav_bytes(self.__audio_data)
        else:
            import soundfile as sf

            audio, sample_rate = sf.read(filepath, dtype="float32")
            # Convert stereo to mono if needed
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        self.__logger.debug(f"Audio decode took {time.monotonic() - t0:.2f}s")

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        audio_duration = len(audio) / sample_rate

        # Run VAD to find speech regions
        t0 = time.monotonic()
        vad_segments = _run_vad(audio, sample_rate)
        self.__logger.debug(f"VAD took {time.monotonic() - t0:.2f}s, found {len(vad_segments)} speech segments")

        if not vad_segments:
            self.__logger.info("No speech detected by VAD")
            return self.__process_transcription([])

        # Transcribe each VAD segment and collect results
        t0 = time.monotonic()
        all_words = []
        all_segments = []
        use_dtw = True  # Try DTW first; disable on failure

        generate_kwargs = {
            "return_timestamps": True,
            "task": "transcribe",
        }
        if self.__language:
            generate_kwargs["language"] = self.__language

        for seg_start, seg_end in vad_segments:
            # Skip very short segments
            if seg_end - seg_start < 0.5:
                continue

            start_sample = int(seg_start * sample_rate)
            end_sample = int(seg_end * sample_rate)
            segment_audio = audio[start_sample:end_sample]

            inputs = self.__processor(
                segment_audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = inputs.input_features.to(
                self.__model.device, dtype=self.__model.dtype
            )
            attention_mask = inputs.attention_mask.to(self.__model.device)

            # Try DTW word-level timestamps
            if use_dtw:
                try:
                    output = self.__model.generate(
                        input_features,
                        attention_mask=attention_mask,
                        return_token_timestamps=True,
                        **generate_kwargs,
                    )
                    words = _tokens_to_words(
                        self.__processor,
                        output["sequences"][0],
                        output["token_timestamps"][0],
                    )
                    for w in words:
                        w["start"] += seg_start
                        w["end"] += seg_start
                        all_words.append(w)
                    continue
                except (RuntimeError, IndexError, KeyError):
                    use_dtw = False
                    self.__logger.info(
                        "DTW token timestamps not supported by this model, "
                        "falling back to segment-level timestamps"
                    )

            # Fallback: segment-level timestamps from timestamp tokens
            try:
                output = self.__model.generate(
                    input_features, attention_mask=attention_mask, **generate_kwargs
                )
            except (RuntimeError, IndexError):
                self.__logger.debug(
                    f"Skipping VAD segment {seg_start:.2f}-{seg_end:.2f}s (generate failed)"
                )
                continue

            segments = _parse_timestamp_tokens(
                self.__processor, output["sequences"][0]
            )
            for seg in segments:
                seg["start"] += seg_start
                seg["end"] += seg_start
                all_segments.append(seg)

        elapsed = time.monotonic() - t0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        self.__logger.info(
            f"Whisper inference took {elapsed:.2f}s for {audio_duration:.1f}s audio ({1/rtf:.1f}x realtime)"
            if rtf > 0 else f"Whisper inference took {elapsed:.2f}s"
        )

        # Combine DTW word-level results and fallback segment-level results
        grouped_segments = _group_words_into_segments(all_words)
        grouped_segments.extend(all_segments)
        grouped_segments.sort(key=lambda s: s["start"])

        return self.__process_transcription(grouped_segments)

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

            chunk_middle = (chunk_start + chunk_end) / 2
            dominant_speaker = self.__get_speaker(diarization, chunk_middle)
            active_speakers = self.__get_speakers_in_range(
                diarization, chunk_start, chunk_end
            )

            aligned_segments.append({
                "start": float(chunk_start),
                "end": float(chunk_end),
                "text": chunk_text.strip(),
                "speaker": dominant_speaker,
                "active_speakers": active_speakers,
                "duration": float(chunk_end - chunk_start),
            })

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

        character = caption[current_position]

        while character != " ":
            if current_position == 0 or len(caption) <= current_position:
                break

            character = caption[current_position]
            current_position -= 1

        first_line = caption[: current_position + 1].strip()
        second_line = caption[current_position + 1 :].strip()

        if len(first_line) <= 1 or len(second_line) <= 1:
            return caption

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
