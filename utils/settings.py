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
