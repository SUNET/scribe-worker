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

import json

from functools import lru_cache
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar
from utils.args import parse_arguments

_, _, _, envfile, _, _, _, _, _, _ = parse_arguments()


class Settings(BaseSettings):
    """
    Settings for the application.
    """

    model_config = SettingsConfigDict(
        env_file=envfile,
        env_file_encoding="utf-8",
        case_sensitive=True,
        validate_assignment=True,
    )

    WORKERS: int = 2
    API_BACKEND_URL: str = ""
    API_VERSION: str = "v1"
    FFMPEG_PATH: str = "ffmpeg"

    HF_TOKEN: str = ""

    # Diarization models, from Hugging Face. Nemotron runs first and pyannote
    # is the fallback, so neither is a Whisper model and they are not in
    # WHISPER_MODELS_HF. NEMOTRON_MODEL can be any model that NeMo's
    # SortformerEncLabelModel loads, and then NEMOTRON_MAX_SPEAKERS must be the
    # most speakers that model tells apart (8 for Nemotron 3). A job with more
    # speakers configured goes straight to pyannote, and so does one where
    # Nemotron reports that many, since it cannot say "more". PYANNOTE_MODEL is
    # gated and needs HF_TOKEN, with its licence accepted.
    NEMOTRON_MODEL: str = "nvidia/Nemotron-3-Diarization"
    NEMOTRON_MAX_SPEAKERS: int = 8
    PYANNOTE_MODEL: str = "pyannote/speaker-diarization-community-1"

    # Ask whisper-timestamped for a per-word confidence score. Word level
    # timestamps are produced either way; this only controls the score, which
    # costs an extra pass over the decoded tokens. Turn off to trade the
    # confidence display in the editor for a slightly faster transcription.
    WORD_CONFIDENCE: bool = True

    # Subtitle shaping. SUBTITLE_LINE_LENGTH should match CHARACTER_LIMIT in
    # scribe-ui: the worker wraps captions at it, the editor flags lines
    # that exceed it, and they will disagree visibly if they drift apart.
    SUBTITLE_LINE_LENGTH: int = 42

    # Segments longer than this many characters are cut in two, at a word
    # boundary when word timings are available.
    SEGMENT_SPLIT_LENGTH: int = 90

    # A silence of at least this many seconds between two words always ends a
    # caption, whatever the length. Keep it at SUBTITLE_MERGE_GAP or above, or
    # the cut is merged straight back into one subtitle.
    SEGMENT_SPLIT_PAUSE: float = 1.8

    # How far whisper-timestamped may move segment boundaries when it refines
    # them, in seconds (a multiple of 0.02). 0 keeps Whisper's own. Refining
    # can tighten timestamps on clean audio but costs time, so measure it on
    # real recordings before raising it.
    WHISPER_REFINE_PRECISION: float = 0.0

    # Two consecutive one-line captions are merged into one two-line subtitle
    # when the silence between them is shorter than this, in seconds.
    SUBTITLE_MERGE_GAP: float = 1.8

    # Start each line with SUBTITLE_SPEAKER_DASH when the two lines of a cue
    # are said by different speakers. This needs diarization, which a job
    # that asks for a transcript runs anyway; for the others it is an extra
    # step, skipped when the job says there is a single speaker. If it fails,
    # the subtitles are made without dashes. Turn off to skip it.
    SUBTITLE_SPEAKER_DASHES: bool = True
    SUBTITLE_SPEAKER_DASH: str = "- "

    # Subtitle timing, in seconds. The limits follow the Netflix timed text
    # style guide, except the reading speed, which is the 18 characters per
    # second scribe-ui flags in "Validate" (MAX_CHARS_PER_SECOND there; keep
    # the two the same): a cue between 5/6 s and 7 s, and 2 frames (at 24 fps)
    # between cues. A cue starts
    # SUBTITLE_LEAD_IN before its first word and stays up SUBTITLE_LEAD_OUT
    # after its last. It is held at least SUBTITLE_MIN_DURATION, and long
    # enough to read at SUBTITLE_MAX_CPS when the next cue leaves room, but
    # never past SUBTITLE_MAX_DURATION. Consecutive cues are kept
    # SUBTITLE_MIN_GAP apart.
    SUBTITLE_LEAD_IN: float = 0.1
    SUBTITLE_LEAD_OUT: float = 0.3
    SUBTITLE_MIN_DURATION: float = 0.833
    SUBTITLE_MAX_CPS: float = 18.0
    SUBTITLE_MAX_DURATION: float = 7.0
    SUBTITLE_MIN_GAP: float = 0.08

    # How far before its first word a cue may start when that is the only way
    # to give it time to read (SUBTITLE_MAX_CPS), in seconds. Beyond
    # SUBTITLE_LEAD_IN this is a compromise between sync and reading speed.
    SUBTITLE_MAX_EARLY: float = 0.2

    # whisper-timestamped marks a word as starting before the audio does, by
    # about 0.1 s on average. Every word start is moved this much later (never
    # past the end of the word), which corrects the word timings and so the
    # subtitles built from them.
    WORD_START_SHIFT: float = 0.1

    # SSL configuration
    SSL_CERTFILE: str = ""
    SSL_KEYFILE: str = ""

    # Path to JSON file with whisper HF models (optional override)
    WHISPER_MODELS_HF_FILE: str = ""

    WHISPER_MODELS_HF: ClassVar[dict[str, dict[str, str]]] = {
        "Swedish": {
            "slower transcription (higher accuracy)": "kblab/kb-whisper-large",
        },
        "Swedish (verbatim)": {
            "slower transcription (higher accuracy)": "kblab/kb-whisper-large@strict",
        },
        "English": {
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "English (verbatim)": {
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "Finnish": {
            "slower transcription (higher accuracy)": "Finnish-NLP/whisper-large-finnish-v3",
        },
        "Danish": {
            "slower transcription (higher accuracy)": "syvai/hviske-v2",
        },
        "Norwegian": {
            "slower transcription (higher accuracy)": "NbAiLabBeta/nb-whisper-large",
        },
        "Norwegian (verbatim)": {
            "slower transcription (higher accuracy)": "NbAiLabBeta/nb-whisper-large-verbatim",
        },
        "Icelandic": {
            "slower transcription (higher accuracy)": "language-and-voice-lab/whisper-large-icelandic-30k-steps-1000h",
        },
        "Northern Sámi (Experimental)": {
            "slower transcription (higher accuracy)": "NbAiLab/whisper-large-sme",
        },
        "French": {
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "German": {
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "Spanish": {
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "Italian": {
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "Russian": {
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "Ukrainian": {
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "Portuguese": {
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
        "Dutch": {
            "slower transcription (higher accuracy)": "openai/whisper-large-v3",
        },
    }

    @model_validator(mode="after")
    def load_whisper_models_hf(self) -> "Settings":
        if not self.WHISPER_MODELS_HF_FILE:
            return self

        path = Path(self.WHISPER_MODELS_HF_FILE)
        if not path.exists():
            return self

        self.__class__.WHISPER_MODELS_HF = json.loads(path.read_text(encoding="utf-8"))
        return self


@lru_cache
def get_settings() -> Settings:
    """
    Get the settings for the application.
    """
    return Settings()
