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
Speaker diarization with NVIDIA Nemotron 3 Diarization (a Sortformer model).

NeMo is imported when the model is first loaded, so it costs nothing unless
it is used.
"""

import logging
import os
import torch

from contextlib import contextmanager, redirect_stderr
from typing import Any, Union

import numpy as np

# Telemetry stays off. Set before NeMo and huggingface_hub are imported.
os.environ["NEMO_ONE_LOGGER_ENABLED"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Sample rate the model expects (the worker decodes to 16 kHz mono).
SAMPLE_RATE = 16000


def download(model: str) -> str:
    """
    Download the model (a Hugging Face repository name, see NEMOTRON_MODEL in
    the settings) into the Hugging Face cache, where load_model() finds it, so
    that the first job does not have to wait for it, or run offline. Only the
    NeMo checkpoint is fetched: the repository can also hold copies of the
    model for other runtimes, several hundred MB each. Returns the path of the
    downloaded files.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["*.nemo"])


def load_model(model: str) -> Any:
    """
    Load the model (a Hugging Face repository name, see NEMOTRON_MODEL in the
    settings), using the offline-style streaming configuration (30.4 s input
    buffer) recommended on the model card.
    """
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.utils import logging as nemo_logging

    # Only errors from NeMo and its OneLogger telemetry library (which is
    # disabled above, but still logs about it).
    nemo_logging.setLevel(logging.ERROR)
    logging.getLogger("nv_one_logger").setLevel(logging.CRITICAL)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    loaded = SortformerEncLabelModel.from_pretrained(model, map_location=device)
    loaded.eval()

    loaded.sortformer_modules.chunk_len = 340
    loaded.sortformer_modules.chunk_right_context = 40
    loaded.sortformer_modules.fifo_len = 40
    loaded.sortformer_modules.spkcache_update_period = 300
    loaded._check_streaming_parameters()

    return loaded


@contextmanager
def quiet_progress_bars():
    """
    Swallow what tqdm writes to stderr inside the block. NeMo creates some of
    its progress bars with an explicit disable argument, which the TQDM_DISABLE
    environment variable cannot override. Real errors are exceptions and are
    unaffected.
    """
    with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
        yield


def parse_segments(raw_segments) -> list:
    """
    Convert the model output into (start, end, speaker) tuples. NeMo returns
    "begin end speaker_N" strings. Speakers are named Speaker_00, Speaker_01,
    ... in order of first arrival.
    """
    segments = []

    for raw in raw_segments:
        start, end, speaker = raw.split()[:3]
        digits = "".join(c for c in speaker if c.isdigit())

        segments.append(
            (float(start), float(end), f"Speaker_{int(digits or 0):02d}")
        )

    return segments


def diarize(
    model: Any, audio: Union[np.ndarray, str], sample_rate: int = SAMPLE_RATE
) -> list:
    """
    Diarize a float32 mono array (at sample_rate) or an audio file path.
    The model detects the number of speakers itself. Returns (start, end,
    speaker) tuples.
    """
    with torch.inference_mode(), quiet_progress_bars():
        predicted = model.diarize(
            audio=[audio], batch_size=1, sample_rate=sample_rate, verbose=False
        )

    return parse_segments(predicted[0])
