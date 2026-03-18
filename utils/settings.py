import json

from functools import lru_cache
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
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
            "slower transcription (higher accuracy)": "nyrahealth/CrisperWhisper",
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
