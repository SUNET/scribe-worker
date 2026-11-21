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

"""
Speaker diarization with pyannote. Used where Nemotron cannot be relied on:
when more speakers are configured than it handles, or when it reports its
maximum and the real count may be higher.

pyannote is imported when the pipeline is first loaded, so it costs nothing
unless it is used.
"""

import os
import torch
import warnings

from typing import Any, Union

import numpy as np

SAMPLE_RATE = 16000


def download(model: str, hf_token: str) -> str:
    """
    Download the pipeline (a Hugging Face repository name, see PYANNOTE_MODEL in
    the settings) into the Hugging Face cache, where load_pipeline() finds it.
    Needs a token whose owner has accepted the model's licence. Returns the path
    of the downloaded files.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model,
        allow_patterns=["config.yaml", "*.bin", "*.npz"],
        token=hf_token,
    )


def load_pipeline(model: str, device: str) -> Any:
    """
    Load the pipeline onto device. Needs a Hugging Face token, in the
    HF_TOKEN environment variable, with the model licence accepted.
    """
    # Before the import, which is where pyannote warns about torchcodec.
    warnings.filterwarnings("ignore", module="pyannote")
    warnings.filterwarnings("ignore", message=".*torchcodec.*")
    # pyannote sends usage telemetry (audio duration, speaker hints) to
    # otel.pyannote.ai by default. Switch it off twice over: its own flag, and
    # the OpenTelemetry SDK that carries it. Both before the import.
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    os.environ["OTEL_SDK_DISABLED"] = "true"

    from pyannote.audio import Pipeline
    from pyannote.audio.telemetry import set_telemetry_metrics

    set_telemetry_metrics(False)

    return Pipeline.from_pretrained(model, token=os.environ.get("HF_TOKEN")).to(
        torch.device(device)
    )


def speaker_hints(speakers: int) -> dict:
    """
    Speaker count hints from the configured number of speakers (0 = detect).
    """
    match speakers:
        case 0:
            return {"num_speakers": None, "min_speakers": None, "max_speakers": None}
        case 1:
            return {"num_speakers": 1, "min_speakers": 1, "max_speakers": 2}
        case _:
            return {
                "num_speakers": speakers,
                "min_speakers": speakers - 1,
                "max_speakers": speakers + 1,
            }


def diarize(
    pipeline: Any,
    audio: Union[np.ndarray, str],
    speakers: int = 0,
    sample_rate: int = SAMPLE_RATE,
) -> list:
    """
    Diarize a float32 mono array (at sample_rate) or an audio file path.
    Returns (start, end, speaker) tuples, with speakers named Speaker_00,
    Speaker_01, ...
    """
    if isinstance(audio, np.ndarray):
        audio = {
            "waveform": torch.from_numpy(audio).unsqueeze(0),
            "sample_rate": sample_rate,
        }

    result = pipeline(audio, **speaker_hints(speakers))

    segments = []

    for turn, _, label in result.speaker_diarization.itertracks(yield_label=True):
        digits = "".join(c for c in str(label) if c.isdigit())

        segments.append(
            (float(turn.start), float(turn.end), f"Speaker_{int(digits or 0):02d}")
        )

    return segments
