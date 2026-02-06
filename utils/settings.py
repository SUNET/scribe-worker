import os

import json
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from typing import ClassVar
from utils.args import parse_arguments

_, _, _, envfile, _, _, _, _ = parse_arguments()


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
    FILE_STORAGE_DIR: str = "/storage"
    API_BACKEND_URL: str = ""
    API_VERSION: str = "v1"
    FFMPEG_PATH: str = "ffmpeg"

    # Ollama configuration
    OLLAMA_URL: str = "http://localhost:11434"

    # HF Whisper configuration.
    HF_WHISPER: bool = False
    HF_TOKEN: str = ""

    # SSL configuration
    SSL_CERTFILE: str = ""
    SSL_KEYFILE: str = ""

    # whisper.cpp path
    WHISPER_CPP_PATH: str = "whisper-cli"

    # Path to JSON file with whisper.cpp models
    WHISPER_MODELS_CPP_FILE: str = "" 

    # Mapping between language and model
    # These are used if no file is supplied
    WHISPER_MODELS_CPP: ClassVar[dict[str, dict[str, str]]] = {
        "Swedish": {
            "fast transcription (normal accuracy)": "sv_base.bin",
            "slower transcription (higher accuracy)": "sv_large.bin",
        },
        "English": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "Finnish": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "Danish": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "dk_large.bin",
        },
        "Norwegian": {
            "fast transcription (normal accuracy)": "no_base.bin",
            "slower transcription (higher accuracy)": "no_large.bin",
        },
        "French": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "German": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "Spanish": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "Italian": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "Russian": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "Ukrainian": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "Portuguese": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
        "Dutch": {
            "fast transcription (normal accuracy)": "whisper_base.bin",
            "slower transcription (higher accuracy)": "whisper_large.bin",
        },
    }

    WHISPER_MODELS_HF: ClassVar[dict[str, dict[str, str]]] = {
        "Swedish": {
            "fast transcription (normal accuracy)": "kblab/kb-whisper-base",
            "slower transcription (higher accuracy)": "kblab/kb-whisper-large",
        },
        "English": {
            "fast transcription (normal accuracy)": "openai/whisper-base-en",
            "slower transcription (higher accuracy)": "openai/whisper-large-en",
        },
        "Finnish": {
            "fast transcription (normal accuracy)": "openai/whisper-base",
            "slower transcription (higher accuracy)": "openai/whisper-large-v2",
        },
        "Danish": {
            "fast transcription (normal accuracy)": "openai/whisper-base",
            "slower transcription (higher accuracy)": "openai/whisper-large-v2",
        },
        "Norwegian": {
            "fast transcription (normal accuracy)": "openai/whisper-base",
            "slower transcription (higher accuracy)": "openai/whisper-large-v2",
        },
    }

    @model_validator(mode="after")
    def load_whisper_models_cpp(self) -> "Settings":
        if not self.WHISPER_MODELS_CPP_FILE:
            return self

        path = Path(self.WHISPER_MODELS_CPP_FILE)
        if not path.exists():
            return self

        # Load cpp model config if available.
        self.__class__.WHISPER_MODELS_CPP = json.loads(
            path.read_text(encoding="utf-8")
        )
        return self


@lru_cache
def get_settings() -> Settings:
    """
    Get the settings for the application.
    """
    if not os.path.exists(Settings().FILE_STORAGE_DIR):
        os.makedirs(Settings().FILE_STORAGE_DIR)

    return Settings()
