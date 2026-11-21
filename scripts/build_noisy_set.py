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

"""
Build a noisy test set with exact reference text, for eval.py.

    uv run python scripts/build_noisy_set.py clean.tsv OUT_DIR \\
        --noise music.mp3 --noise pink --snrs 10,0

clean.tsv has one "audio path<TAB>reference text" line per clean clip (see
fetch_fleurs.py). For each condition one long recording is made:

    [noise only] clip 1 [noise only] clip 2 [noise only] ... clip N [noise only]

with a continuous noise track under it at the given speech-to-noise ratio.
The reference is the clips' text in order, and the noise-only stretches have
none, so every word Whisper writes there is invented. The same clips, order
and noise are used for every condition, so the results are paired. A "clean"
condition (silence between the clips) is always included, and the others are
named NOISE-snrDB, for example music-snr10.

--noise may be repeated and takes a media file (music, a recording of a room:
anything ffmpeg reads, ideally one without speech), or a built-in source:
    pink     pink noise
    babble   the clips themselves overlapped, a crowd of voices. Note that
             Whisper may truly transcribe these, so it is not a clean
             measure of invented words.

Writes OUT_DIR/*.wav and OUT_DIR/set.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evallib import SAMPLE_RATE, load_audio, log, write_wav


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0


def fit_length(audio: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """A stretch of `length` samples from audio, looped if it is too short."""
    if len(audio) == 0:
        raise SystemExit("the noise source is empty")

    if len(audio) < length:
        audio = np.tile(audio, int(np.ceil(length / len(audio))))

    start = int(rng.integers(0, len(audio) - length + 1))

    return audio[start : start + length]


def pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(length))
    frequencies = np.maximum(np.arange(len(spectrum)), 1)

    return np.fft.irfft(spectrum / np.sqrt(frequencies), n=length).astype(np.float32)


def babble(clips: list, length: int, rng: np.random.Generator) -> np.ndarray:
    mixed = np.zeros(length, dtype=np.float32)

    for index in rng.choice(len(clips), size=min(6, len(clips)), replace=False):
        mixed += fit_length(clips[index], length, rng)

    return mixed


def noise_name(source: str) -> str:
    return source if source in ("pink", "babble") else Path(source).stem


def build_set(
    clean_tsv: Path,
    out_dir: Path,
    noises: list,
    snrs: list,
    gap_seconds: float = 8.0,
    seed: int = 1,
) -> Path:
    """Write the recordings and set.json to out_dir. Returns set.json."""

    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []

    for line in clean_tsv.read_text(encoding="utf-8").splitlines():
        if line.strip():
            path, text = line.split("\t", 1)
            entries.append((load_audio(path), text.strip()))

    if not entries:
        raise SystemExit(f"no clips in {clean_tsv}")

    clips = [audio for audio, _ in entries]
    gap = int(gap_seconds * SAMPLE_RATE)

    # One layout for every condition: the speech track, and where things are.
    speech = [np.zeros(gap, dtype=np.float32)]
    speech_spans, gap_spans, cursor = [], [(0.0, gap_seconds)], gap

    for audio, text in entries:
        speech.append(audio)
        speech_spans.append(
            {"start": cursor / SAMPLE_RATE, "end": (cursor + len(audio)) / SAMPLE_RATE, "text": text}
        )
        cursor += len(audio)
        speech.append(np.zeros(gap, dtype=np.float32))
        gap_spans.append((cursor / SAMPLE_RATE, (cursor + gap) / SAMPLE_RATE))
        cursor += gap

    track = np.concatenate(speech)
    active = np.concatenate(clips)
    reference = " ".join(text for _, text in entries)

    conditions = [("clean", None, None)]

    for source in noises:
        if source == "pink":
            noise = pink_noise(len(track), rng)
        elif source == "babble":
            noise = babble(clips, len(track), rng)
        else:
            noise = fit_length(load_audio(source), len(track), rng)

        conditions += [(f"{noise_name(source)}-snr{snr:g}", snr, noise) for snr in snrs]

    files = []

    for name, snr, noise in conditions:
        mixed = track.copy()

        if noise is not None:
            gain = rms(active) / (10 ** (snr / 20)) / max(rms(noise), 1e-9)
            mixed = mixed + noise * gain

        peak = float(np.max(np.abs(mixed)))
        if peak > 0.95:
            mixed = mixed * (0.95 / peak)

        target = out_dir / f"{name}.wav"
        write_wav(target, mixed)
        files.append(
            {
                "condition": name,
                "snr": snr,
                "audio": str(target.resolve()),
                "reference": reference,
                "speech": speech_spans,
                "gaps": gap_spans,
            }
        )

    manifest = out_dir / "set.json"
    manifest.write_text(json.dumps({"files": files}, indent=1), encoding="utf-8")
    log(f"{len(files)} recordings of {len(track) / SAMPLE_RATE:.0f} s in {out_dir}")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("clean_tsv")
    parser.add_argument("out_dir")
    parser.add_argument("--noise", action="append", default=[], help="media file, pink or babble; repeatable")
    parser.add_argument("--snrs", default="10,0", help="speech to noise, in dB")
    parser.add_argument("--gap", type=float, default=8.0, help="noise-only seconds between clips")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    build_set(
        Path(args.clean_tsv),
        Path(args.out_dir),
        args.noise or ["pink"],
        [float(s) for s in args.snrs.split(",")],
        args.gap,
        args.seed,
    )


if __name__ == "__main__":
    main()
