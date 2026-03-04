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

import os

import json
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from typing import ClassVar
from utils.args import parse_arguments

_, _, _, envfile, _, _, _ = parse_arguments()


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

    DEBUG: bool = True
    WORKERS: int = 2
    FILE_STORAGE_DIR: str = "./storage"
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
            "fast transcription (normal accuracy)": "kblab/kb-whisper-base",
            "slower transcription (higher accuracy)": "kblab/kb-whisper-large",
        },
        "English": {
            "fast transcription (normal accuracy)": "base.en",
            "slower transcription (higher accuracy)": "large-v3",
        },
        "Finnish": {
            "fast transcription (normal accuracy)": "base",
            "slower transcription (higher accuracy)": "large-v3",
        },
        "Danish": {
            "fast transcription (normal accuracy)": "base",
            "slower transcription (higher accuracy)": "large-v3",
        },
        "Norwegian": {
            "fast transcription (normal accuracy)": "base",
            "slower transcription (higher accuracy)": "large-v3",
        },
        "French": {
            "fast transcription (normal accuracy)": "base",
            "slower transcription (higher accuracy)": "large-v3",
        },
        "German": {
            "fast transcription (normal accuracy)": "base",
            "slower transcription (higher accuracy)": "large-v3",
        },
        "Spanish": {
            "fast transcription (normal accuracy)": "base",
            "slower transcription (higher accuracy)": "large-v3",
        },
        "Italian": {
            "fast transcription (normal accuracy)": "base",
            "slower transcription (higher accuracy)": "large-v3",
        },
        "Russian": {
            "fast transcription (normal accuracy)": "base",
            "slower transcription (higher accuracy)": "large-v3",
        },
        "Ukrainian": {
            "fast transcription (normal accuracy)": "base",
            "slower transcription (higher accuracy)": "large-v3",
        },
        "Portuguese": {
            "fast transcription (normal accuracy)": "base",
            "slower transcription (higher accuracy)": "large-v3",
        },
        "Dutch": {
            "fast transcription (normal accuracy)": "base",
            "slower transcription (higher accuracy)": "large-v3",
        },
    }

    @model_validator(mode="after")
    def load_whisper_models_hf(self) -> "Settings":
        if not self.WHISPER_MODELS_HF_FILE:
            return self

        path = Path(self.WHISPER_MODELS_HF_FILE)
        if not path.exists():
            return self

        self.__class__.WHISPER_MODELS_HF = json.loads(
            path.read_text(encoding="utf-8")
        )
        return self


@lru_cache
def get_settings() -> Settings:
    """
    Get the settings for the application.
    """
    return Settings()
