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
Compare transcription engines on one recording with a reference SRT.

    uv run --with faster-whisper python scripts/benchmark_engines.py \\
        AUDIO.mp4 REFERENCE.srt --model kblab/kb-whisper-large \\
        --language swedish --engines current,faster:float32,faster:int8

Engines:
    current            the production path (whisper-timestamped via
                       WhisperAudioTranscriber)
    hf                 transformers pipeline on the same safetensors
                       (segment timings only)
    faster[:TYPE]      faster-whisper, TYPE is the CTranslate2 compute type
                       (default float32 on cpu, float16 on cuda)

Run from the repository root so .env is found. Reports, per engine:
    - WER against the reference SRT (and substitutions/insertions/deletions)
    - caption boundary error against the reference SRT timings
    - word timing difference between engines, on words both agree on
    - confidence distribution, and how well it separates right from wrong words
    - model load time and inference speed

Caveats: the alignment is word level Levenshtein, so it is meant for short
files (it is quadratic). Reference timings are only as good as the SRT. If the
reference was produced by one of the engines it favours that engine.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

# Same decoding settings as utils/whisper.py, so engines are compared on
# equal terms rather than on each library's defaults.
BEAM_SIZE = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("audio")
    parser.add_argument(
        "reference_srt",
        nargs="?",
        default="",
        help="Optional. Without it only engine-vs-engine comparison is made.",
    )
    parser.add_argument("--model", default="kblab/kb-whisper-large")
    parser.add_argument("--language", default="swedish")
    parser.add_argument("--engines", default="current,faster")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--out", default="benchmark_out")
    parser.add_argument(
        "--fw-device",
        default="auto",
        help="faster-whisper device: auto, cpu or cuda (no MPS support).",
    )
    parser.add_argument(
        "--fw-model",
        default="",
        help="faster-whisper model directory or id, when it should differ from "
        "--model. KBLab's published CTranslate2 model.bin can be older than "
        "their safetensors, so convert the same checkpoint with "
        "ct2-transformers-converter to compare engines on equal weights.",
    )
    parser.add_argument(
        "--fw-no-vad",
        action="store_true",
        help="Turn off faster-whisper's VAD filter.",
    )
    parser.add_argument(
        "--current-no-vad",
        action="store_true",
        help="Run the current engine with whisper-timestamped's vad=False.",
    )
    parser.add_argument(
        "--fw-vad-min-silence-ms",
        type=int,
        default=None,
        help="faster-whisper VAD min_silence_duration_ms (library default 2000).",
    )
    parser.add_argument(
        "--fw-vad-speech-pad-ms",
        type=int,
        default=None,
        help="faster-whisper VAD speech_pad_ms (library default 400).",
    )
    parser.add_argument(
        "--fw-fallback",
        action="store_true",
        help="Enable faster-whisper's temperature fallback "
        "(current engine uses temperature 0 only).",
    )
    parser.add_argument("--show-errors", type=int, default=25)

    return parser.parse_args()


# ---------------------------------------------------------------- audio, srt


def load_audio(path: str, ffmpeg: str) -> np.ndarray:
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-i", path, "-vn", "-ar", str(SAMPLE_RATE), "-ac", "1"]
        + ["-f", "s16le", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace"))

    pcm = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)

    return pcm / 32768.0


def pcm_to_wav_bytes(audio: np.ndarray) -> bytes:
    import io
    import wave

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((audio * 32767.0).astype(np.int16).tobytes())

    return buffer.getvalue()


def srt_time(value: str) -> float:
    hours, minutes, rest = value.strip().split(":")
    seconds, millis = rest.replace(".", ",").split(",")

    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def parse_srt(path: str) -> list:
    """
    Returns [{"start", "end", "text"}].
    """

    captions = []
    text = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")

    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.split("\n")
        timing = next((i for i, line in enumerate(lines) if "-->" in line), None)

        if timing is None:
            continue

        start, end = lines[timing].split("-->")
        body = " ".join(line.strip() for line in lines[timing + 1 :]).strip()

        captions.append({"start": srt_time(start), "end": srt_time(end), "text": body})

    return captions


# ------------------------------------------------------------------- engines


def run_current(audio: np.ndarray, args: argparse.Namespace) -> dict:
    from utils.whisper import WhisperAudioTranscriber

    if args.current_no_vad:
        import utils.whisper as engine

        original = engine.whisper.transcribe

        def transcribe_without_vad(*a, **kw):
            kw["vad"] = False
            return original(*a, **kw)

        engine.whisper.transcribe = transcribe_without_vad

    t0 = time.monotonic()
    transcriber = WhisperAudioTranscriber(
        model_name=args.model,
        language=args.language,
        audio_data=pcm_to_wav_bytes(audio),
    )
    load_seconds = time.monotonic() - t0

    t0 = time.monotonic()
    transcribed = transcriber.transcribe()
    inference_seconds = time.monotonic() - t0

    if transcribed is None:
        raise RuntimeError("current engine failed, see log")

    payload = transcriber.words() or {"words": []}

    return {
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "text": getattr(transcriber, "_WhisperAudioTranscriber__full_transcription"),
        "words": payload["words"],
        "segments": [
            {"start": c["start"], "end": c["end"], "text": c["text"]}
            for c in getattr(transcriber, "_WhisperAudioTranscriber__chunks")
        ],
        "settings": "whisper-timestamped, "
        + ("vad=False" if args.current_no_vad else "as in production"),
    }


def run_faster(audio: np.ndarray, args: argparse.Namespace, compute_type: str) -> dict:
    from faster_whisper import WhisperModel

    model_name, revision = args.fw_model or args.model, None

    if "@" in model_name and not Path(model_name).exists():
        model_name, revision = model_name.rsplit("@", 1)

    device = args.fw_device

    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    if not compute_type:
        compute_type = "float16" if device == "cuda" else "float32"

    t0 = time.monotonic()
    model = WhisperModel(
        model_name, device=device, compute_type=compute_type, revision=revision
    )
    load_seconds = time.monotonic() - t0

    language = args.language.split("(")[0].strip().lower()
    language_code = {"swedish": "sv", "english": "en", "norwegian": "no"}.get(
        language, language
    )

    options = {
        "language": language_code,
        "beam_size": BEAM_SIZE,
        "condition_on_previous_text": False,
        "vad_filter": not args.fw_no_vad,
        "word_timestamps": True,
    }

    vad_parameters = {}

    if args.fw_vad_min_silence_ms is not None:
        vad_parameters["min_silence_duration_ms"] = args.fw_vad_min_silence_ms

    if args.fw_vad_speech_pad_ms is not None:
        vad_parameters["speech_pad_ms"] = args.fw_vad_speech_pad_ms

    if vad_parameters:
        options["vad_parameters"] = vad_parameters

    if not args.fw_fallback:
        options["temperature"] = 0.0

    t0 = time.monotonic()
    segments, _ = model.transcribe(audio, **options)
    segments = list(segments)  # transcription is lazy, force it inside the timer
    inference_seconds = time.monotonic() - t0

    words = []

    for segment in segments:
        for word in segment.words or []:
            text = word.word.strip()

            if text:
                words.append(
                    {
                        "t": text,
                        "s": round(word.start, 3),
                        "e": round(word.end, 3),
                        "c": round(word.probability, 3),
                    }
                )

    return {
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "text": " ".join(s.text.strip() for s in segments if s.text.strip()),
        "words": words,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments
        ],
        "settings": f"faster-whisper model={model_name} device={device} "
        f"compute_type={compute_type} vad={'off' if args.fw_no_vad else 'on'} "
        f"fallback={'on' if args.fw_fallback else 'off'}",
    }


def run_hf(audio: np.ndarray, args: argparse.Namespace) -> dict:
    """
    Plain transformers pipeline on the same safetensors: a tie-breaker that
    uses neither whisper-timestamped nor CTranslate2. Segment level only, so
    word timing metrics are skipped for it.
    """

    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    device = "cuda:0" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    # float16 on MPS produced NaNs (output was all "!"), so stay in float32.
    dtype = torch.float32
    name = args.model.split("@")[0]

    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name, dtype=dtype).to(device)
    load_seconds = time.monotonic() - t0

    language = args.language.split("(")[0].strip().lower()

    t0 = time.monotonic()
    # Whole recording in one go: transformers' sequential long-form decoding
    # (like openai-whisper) slides the 30 s window along the real timestamps.
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
    ).to(device)
    inputs["input_features"] = inputs["input_features"].to(dtype)

    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            language=language,
            task="transcribe",
            num_beams=BEAM_SIZE,
            do_sample=False,
            return_timestamps=True,
            condition_on_prev_tokens=False,
        )

    inference_seconds = time.monotonic() - t0
    text = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    return {
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "text": text,
        "words": [],
        "segments": [],
        "settings": f"transformers generate device={device} dtype={dtype}",
    }


# ------------------------------------------------------------------- metrics


def normalize_token(text: str) -> str:
    """
    Lowercase and drop punctuation. Hyphens vanish, so "e-post" and "epost"
    compare equal; \\w keeps å, ä, ö.
    """

    return re.sub(r"[^\w]", "", text.lower())


def align(reference: list, hypothesis: list) -> list:
    """
    Word level Levenshtein alignment. Returns ops:
    ("eq"|"sub"|"del"|"ins", ref_index|None, hyp_index|None).
    """

    n, m = len(reference), len(hypothesis)
    cost = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost[:, 0] = np.arange(n + 1)
    cost[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(
                cost[i - 1, j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                cost[i - 1, j] + 1,
                cost[i, j - 1] + 1,
            )

    ops = []
    i, j = n, m

    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and cost[i, j]
            == cost[i - 1, j - 1] + (reference[i - 1] != hypothesis[j - 1])
        ):
            ops.append(("eq" if reference[i - 1] == hypothesis[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i, j] == cost[i - 1, j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1

    return ops[::-1]


def summarize(values: list) -> dict:
    if not values:
        return {"n": 0}

    ordered = sorted(values)

    return {
        "n": len(values),
        "mean": round(float(statistics.fmean(values)), 3),
        "median": round(float(statistics.median(values)), 3),
        "p90": round(float(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]), 3),
    }


def reference_tokens(captions: list) -> list:
    """
    [(normalized token, caption index)] for the reference.
    """

    tokens = []

    for index, caption in enumerate(captions):
        for raw in caption["text"].split():
            token = normalize_token(raw)

            if token:
                tokens.append((token, index, raw))

    return tokens


def hypothesis_tokens(result: dict) -> list:
    """
    [(normalized token, word dict or None, raw)]. Uses the timed words when
    present, otherwise the plain text without timings.
    """

    tokens = []

    if result["words"]:
        for word in result["words"]:
            token = normalize_token(word["t"])

            if token:
                tokens.append((token, word, word["t"]))
    else:
        for raw in result["text"].split():
            token = normalize_token(raw)

            if token:
                tokens.append((token, None, raw))

    return tokens


def score_against_reference(result: dict, captions: list, show: int) -> dict:
    ref = reference_tokens(captions)
    hyp = hypothesis_tokens(result)
    ops = align([r[0] for r in ref], [h[0] for h in hyp])

    counts = {"eq": 0, "sub": 0, "del": 0, "ins": 0}

    for op, _, _ in ops:
        counts[op] += 1

    wer = (counts["sub"] + counts["del"] + counts["ins"]) / max(len(ref), 1)

    # Caption boundary error: first/last word of each reference caption that
    # the engine got exactly right, compared with the SRT's own times.
    matched_by_caption = {}

    for op, ref_index, hyp_index in ops:
        word = hyp[hyp_index][1] if hyp_index is not None else None

        if op == "eq" and word is not None:
            matched_by_caption.setdefault(ref[ref_index][1], []).append(word)

    start_errors, end_errors = [], []

    for caption_index, matched in matched_by_caption.items():
        caption = captions[caption_index]
        start_errors.append(matched[0]["s"] - caption["start"])
        end_errors.append(matched[-1]["e"] - caption["end"])

    # Does confidence separate correct words from wrong ones?
    right = [hyp[j][1]["c"] for op, _, j in ops if op == "eq" and hyp[j][1] and "c" in hyp[j][1]]
    wrong = [hyp[j][1]["c"] for op, _, j in ops if op == "sub" and hyp[j][1] and "c" in hyp[j][1]]

    errors = []

    for op, i, j in ops:
        if op != "eq":
            errors.append(
                f"{op:3} ref={ref[i][2] if i is not None else '-':<18} "
                f"hyp={hyp[j][2] if j is not None else '-'}"
                + (
                    f"  (t={hyp[j][1]['s']:.2f}, c={hyp[j][1].get('c', '-')})"
                    if j is not None and hyp[j][1]
                    else ""
                )
            )

    return {
        "reference_words": len(ref),
        "hypothesis_words": len(hyp),
        "wer": round(wer, 4),
        **counts,
        "caption_start_error_s": summarize([abs(e) for e in start_errors]),
        "caption_end_error_s": summarize([abs(e) for e in end_errors]),
        "caption_start_bias_s": round(statistics.fmean(start_errors), 3) if start_errors else None,
        "caption_end_bias_s": round(statistics.fmean(end_errors), 3) if end_errors else None,
        "confidence_all": summarize([h[1]["c"] for h in hyp if h[1] and "c" in h[1]]),
        "confidence_correct": summarize(right),
        "confidence_substituted": summarize(wrong),
        "errors": errors[:show],
    }


def compare_engines(first: dict, second: dict) -> dict:
    """
    Word timing difference on the words both engines produced identically.
    """

    a = hypothesis_tokens(first)
    b = hypothesis_tokens(second)
    ops = align([t[0] for t in a], [t[0] for t in b])

    start_deltas, end_deltas = [], []
    equal = 0

    for op, i, j in ops:
        if op != "eq":
            continue

        equal += 1

        if a[i][1] and b[j][1]:
            start_deltas.append(abs(a[i][1]["s"] - b[j][1]["s"]))
            end_deltas.append(abs(a[i][1]["e"] - b[j][1]["e"]))

    return {
        "text_difference_wer": round(1 - equal / max(len(a), 1), 4),
        "start_delta_s": summarize(start_deltas),
        "end_delta_s": summarize(end_deltas),
    }


# ---------------------------------------------------------------------- main


def main() -> None:
    args = parse_args()

    # utils.* parses sys.argv when imported; hide this script's arguments.
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    captions = parse_srt(args.reference_srt) if args.reference_srt else []
    audio = load_audio(args.audio, args.ffmpeg)
    duration = len(audio) / SAMPLE_RATE
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"audio {duration:.1f}s, reference {len(captions)} captions\n")

    results = {}

    for spec in [e.strip() for e in args.engines.split(",") if e.strip()]:
        name, _, option = spec.partition(":")
        print(f"== {spec}")

        if name == "current":
            result = run_current(audio, args)
        elif name == "hf":
            result = run_hf(audio, args)
        elif name == "faster":
            result = run_faster(audio, args, option)
        else:
            raise SystemExit(f"unknown engine {name!r}")

        result["score"] = (
            score_against_reference(result, captions, args.show_errors)
            if captions
            else None
        )
        result["realtime_factor"] = round(duration / result["inference_seconds"], 2)
        results[spec] = result

        (out_dir / f"{spec.replace(':', '_')}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        score = result["score"]
        print(f"   {result['settings']}")
        print(
            f"   load {result['load_seconds']:.1f}s  inference "
            f"{result['inference_seconds']:.1f}s ({result['realtime_factor']}x realtime)"
        )

        if score is None:
            print(f"   {len(hypothesis_tokens(result))} words produced\n")
            continue

        print(
            f"   WER {score['wer']:.2%}  sub {score['sub']} del {score['del']} "
            f"ins {score['ins']}  of {score['reference_words']} reference words; "
            f"engine produced {score['hypothesis_words']} words"
        )
        print(
            f"   caption start err {score['caption_start_error_s']}\n"
            f"   caption end err   {score['caption_end_error_s']}"
        )
        print(
            f"   confidence all {score['confidence_all']}\n"
            f"     correct     {score['confidence_correct']}\n"
            f"     substituted {score['confidence_substituted']}"
        )

        for line in score["errors"]:
            print(f"     {line}")

        print()

    names = list(results)

    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            comparison = compare_engines(results[first], results[second])
            print(f"== {first} vs {second}: {comparison}")

    (out_dir / "summary.json").write_text(
        json.dumps(
            {name: {k: v for k, v in r.items() if k not in ("words", "segments", "text")} for name, r in results.items()},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nfull output in {out_dir}/")


if __name__ == "__main__":
    main()
