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
Compare Nemotron 3 Diarization (current) with pyannote community-1 (old).

The two engines need incompatible environments, so each runs on its own and
writes an RTTM file; a third step scores the RTTMs.

    # 1. Nemotron, in the project environment (run from the repo root)
    uv run python scripts/benchmark_diarization.py run nemotron AUDIO out/nemotron.rttm

    # 2. pyannote, in a throwaway environment (needs HF_TOKEN, and the
    #    community-1 licence accepted on huggingface.co)
    HF_TOKEN=... uv run --no-project --with pyannote-audio \\
        python scripts/benchmark_diarization.py run pyannote AUDIO out/pyannote.rttm

    # 3. Score. REFERENCE.rttm is optional but strongly recommended.
    uv run --no-project --with pyannote.metrics \\
        python scripts/benchmark_diarization.py score out/nemotron.rttm \\
        out/pyannote.rttm --reference REFERENCE.rttm

`run` also writes <out>.json with model load time, inference time and the
real-time factor. Use --speakers N with pyannote to reproduce the production
hint (Nemotron ignores it). `score` reports, per engine, DER (missed speech,
false alarm, speaker confusion) and detected speaker count against the
reference, plus the DER between the two engines. Without a reference only the
engine-to-engine figures are meaningful: they show how much the engines
disagree, not which one is right.

Use recordings that look like production (noisy, overlapping, many speakers),
not clean podcasts, and include some with more than 4 speakers.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SAMPLE_RATE = 16000


def load_audio(path: str):
    """Decode any media file to 16 kHz mono float32, as the worker does."""
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True,
        check=True,
    ).stdout

    return np.frombuffer(raw, dtype=np.float32).copy()


def write_rttm(path: Path, uri: str, segments: list) -> None:
    lines = [
        f"SPEAKER {uri} 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>"
        for start, end, speaker in sorted(segments)
        if end > start
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_nemotron(audio, speakers, name):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # utils.settings parses sys.argv with the worker's own parser on import.
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        from utils import nemotron
        from utils.settings import get_settings
    finally:
        sys.argv = argv

    t0 = time.monotonic()
    model = nemotron.load_model(name or get_settings().NEMOTRON_MODEL)
    load = time.monotonic() - t0

    t0 = time.monotonic()
    segments = nemotron.diarize(model, audio, SAMPLE_RATE)
    return segments, load, time.monotonic() - t0


def run_pyannote(audio, speakers, name):
    import os

    import torch

    # pyannote sends usage telemetry (audio duration, speaker hints) to
    # otel.pyannote.ai unless told not to. Off before it is imported.
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    os.environ["OTEL_SDK_DISABLED"] = "true"

    from pyannote.audio import Pipeline
    from pyannote.audio.telemetry import set_telemetry_metrics

    set_telemetry_metrics(False)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    t0 = time.monotonic()
    pipeline = Pipeline.from_pretrained(
        name or "pyannote/speaker-diarization-community-1",
        token=os.environ["HF_TOKEN"],
    ).to(device)
    load = time.monotonic() - t0

    # Same hints as the old production code.
    if speakers == 0:
        hints = {"num_speakers": None, "min_speakers": None, "max_speakers": None}
    elif speakers == 1:
        hints = {"num_speakers": 1, "min_speakers": 1, "max_speakers": 2}
    else:
        hints = {
            "num_speakers": speakers,
            "min_speakers": speakers - 1,
            "max_speakers": speakers + 1,
        }

    waveform = torch.from_numpy(audio).unsqueeze(0)

    t0 = time.monotonic()
    result = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **hints)
    elapsed = time.monotonic() - t0

    segments = [
        (turn.start, turn.end, label)
        for turn, _, label in result.speaker_diarization.itertracks(yield_label=True)
    ]

    return segments, load, elapsed


def cmd_run(args) -> None:
    audio = load_audio(args.audio)
    seconds = len(audio) / SAMPLE_RATE

    runner = run_nemotron if args.engine == "nemotron" else run_pyannote
    segments, load, inference = runner(audio, args.speakers, args.model)

    out = Path(args.out)
    write_rttm(out, Path(args.audio).stem, segments)

    stats = {
        "engine": args.engine,
        "audio_seconds": round(seconds, 1),
        "load_seconds": round(load, 2),
        "inference_seconds": round(inference, 2),
        "realtime_factor": round(inference / seconds, 3) if seconds else None,
        "speakers": len({s for _, _, s in segments}),
        "segments": len(segments),
    }
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


def cmd_score(args) -> None:
    from pyannote.database.util import load_rttm
    from pyannote.metrics.diarization import DiarizationErrorRate

    def load(path):
        annotations = load_rttm(path)
        if len(annotations) != 1:
            sys.exit(f"{path}: expected one recording, found {len(annotations)}")
        return next(iter(annotations.values()))

    def report(name, reference, hypothesis):
        metric = DiarizationErrorRate(collar=args.collar, skip_overlap=False)
        detail = metric(reference, hypothesis, detailed=True)
        total = detail["total"] or 1.0
        print(
            f"{name:<28} DER {detail['diarization error rate']:6.1%}  "
            f"missed {detail['missed detection'] / total:6.1%}  "
            f"false alarm {detail['false alarm'] / total:6.1%}  "
            f"confusion {detail['confusion'] / total:6.1%}  "
            f"speakers {len(hypothesis.labels())} vs {len(reference.labels())}"
        )

    hypotheses = {Path(p).stem: load(p) for p in args.rttm}

    print(f"collar {args.collar}s, overlapped speech scored\n")

    for path in args.rttm:
        sidecar = Path(path).with_suffix(".json")
        if sidecar.exists():
            s = json.loads(sidecar.read_text())
            print(
                f"{s['engine']:<10} load {s['load_seconds']}s  "
                f"inference {s['inference_seconds']}s  "
                f"RTF {s['realtime_factor']}  segments {s['segments']}"
            )

    print()

    if args.reference:
        reference = load(args.reference)
        for name, hypothesis in hypotheses.items():
            report(f"{name} vs reference", reference, hypothesis)
        print()

    names = list(hypotheses)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            report(f"{b} vs {a} (disagreement)", hypotheses[a], hypotheses[b])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="diarize one recording with one engine")
    run.add_argument("engine", choices=["nemotron", "pyannote"])
    run.add_argument("audio")
    run.add_argument("out", help="RTTM file to write")
    run.add_argument(
        "--speakers", type=int, default=0,
        help="speaker count hint, pyannote only (0 = detect)",
    )
    run.add_argument(
        "--model",
        help="model to use, a Hugging Face repository name "
        "(default: the one in the settings for nemotron, community-1 for pyannote)",
    )
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="score RTTM files")
    score.add_argument("rttm", nargs="+")
    score.add_argument("--reference", help="ground-truth RTTM")
    score.add_argument("--collar", type=float, default=0.25)
    score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
