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

_, _, _, envfile, _, _, _, _, _, _, _ = parse_arguments()


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

    # Ask whisper-timestamped for a per-word confidence score. Word level
    # timestamps are produced either way; this only controls the score, which
    # costs an extra pass over the decoded tokens. Turn off to trade the
    # confidence display in the editor for a slightly faster transcription.
    WORD_CONFIDENCE: bool = True

    # Subtitle shaping. SUBTITLE_LINE_LENGTH should match CHARACTER_LIMIT in
    # transcribe-ui: the worker wraps captions at it, the editor flags lines
    # that exceed it, and they will disagree visibly if they drift apart.
    SUBTITLE_LINE_LENGTH: int = 42

    # Segments longer than this many characters are cut in two, at a word
    # boundary when word timings are available.
    SEGMENT_SPLIT_LENGTH: int = 90

    # Two consecutive one-line captions are merged into one two-line subtitle
    # when the silence between them is shorter than this, in seconds.
    SUBTITLE_MERGE_GAP: float = 1.8

    # SSL configuration
    SSL_CERTFILE: str = ""
    SSL_KEYFILE: str = ""

    # ------------------------------------------------------------------
    # Inference (--role inference)
    # ------------------------------------------------------------------

    # Websocket the inference hub listens on. The worker connects out to
    # it, presenting the same client certificate the REST calls use, and
    # keeps the connection open waiting for work -- there is no queue to
    # poll, because inference requests are never stored anywhere.
    INFERENCE_HUB_URL: str = ""

    # How this worker names itself to the hub. Empty means the hostname,
    # with the GPU index appended when the host has more than one.
    INFERENCE_WORKER_ID: str = ""

    # How many requests this worker will take at once. One by default: a
    # second concurrent generation on the same GPU makes both slower and
    # risks running out of VRAM mid-answer.
    INFERENCE_CONCURRENCY: int = 1

    # Load the models at startup rather than on the first request. Right in
    # production -- the first reader should not wait through a load of tens
    # of gigabytes -- and usually wrong on a development box, where it means
    # downloading a model before finding out the hub is not even running.
    INFERENCE_PRELOAD: bool = True

    # Seconds between reconnection attempts when the hub is unreachable,
    # and how often to tell it we are still here.
    INFERENCE_RECONNECT_SECONDS: int = 5
    INFERENCE_HEARTBEAT_SECONDS: int = 20

    # Refuse to talk to a model server that is not on this machine or a
    # private network. Transcripts are the whole point of the service being
    # self-hosted; a model endpoint on the public internet would send them
    # somewhere else entirely. Only turn this off with a very good reason.
    INFERENCE_ALLOW_REMOTE: bool = False

    # Model registry. The alias on the left is what the frontend shows and
    # what usage rows record; everything else is deployment detail. Adding a
    # model to compare against is meant to be a configuration change and a
    # restart, never a code change.
    #
    #   provider "transformers"  loads the model into this process and keeps
    #                            it resident. model_id is a Hugging Face repo.
    #   provider "openai"        talks to a local OpenAI-compatible server
    #                            (vLLM, llama.cpp, Ollama). base_url is
    #                            required and must be local unless
    #                            INFERENCE_ALLOW_REMOTE is set.
    #
    # The default below is the production one. Replace the whole map from
    # .env with a JSON object, which python-dotenv reads across lines as
    # long as it is quoted:
    #
    #   INFERENCE_MODELS='{"gemma": {"provider": "openai",
    #                                "model_id": "gemma3:12b",
    #                                "base_url": "http://127.0.0.1:11434/v1"}}'
    #
    # It is validated on the way in, so a typo is a startup error naming the
    # alias rather than a request failing much later with the reader
    # watching.
    INFERENCE_MODELS: dict[str, dict] = {
        "gemma": {
            "provider": "transformers",
            "model_id": "google/gemma-3-27b-it",
            "temperature": 0.3,
            "top_p": 0.95,
        },
    }

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
    def check_inference_models(self) -> "Settings":
        """
        Refuse a model registry that cannot work, at startup.

        The alternative is a reader clicking Analyse, waiting for a worker,
        and being told the model is unusable -- by which time whoever wrote
        the entry has gone home.
        """

        for alias, config in self.INFERENCE_MODELS.items():
            if not isinstance(config, dict):
                raise ValueError(f"INFERENCE_MODELS[{alias}] is not an object.")

            provider = config.get("provider", "")

            if provider not in ("transformers", "openai"):
                raise ValueError(
                    f"INFERENCE_MODELS[{alias}] has provider {provider!r}; "
                    "expected 'transformers' or 'openai'."
                )

            if not config.get("model_id"):
                raise ValueError(f"INFERENCE_MODELS[{alias}] has no model_id.")

            if provider == "openai" and not config.get("base_url"):
                raise ValueError(
                    f"INFERENCE_MODELS[{alias}] uses the openai provider and "
                    "so needs a base_url."
                )

        return self

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
