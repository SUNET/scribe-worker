# scribe-worker

Worker for the Sunet transcription service (Sunet Scribe).

## Author

This project is developed by [Sunet](https://www.sunet.se). Contributor: Kristofer Hallin.

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

Copyright (c) 2025-2026 Sunet. Contributor: Kristofer Hallin.

## Third-party Notices

This repository includes a `NOTICE` file containing attribution
information for third-party components.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## Features

- **Transcription Processing**: Processes audio/video transcription jobs from the backend queue
- **Whisper Transcription**: Runs Whisper locally (through
  [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)) with Hugging Face
  models, and produces per-word timings and confidence scores
- **Speaker Diarization**: Works out who speaks when, with NVIDIA Nemotron 3 Diarization and a
  pyannote fallback (see [Speaker diarization](#speaker-diarization))
- **Multiple Output Formats**: Generates SRT subtitles, a JSON transcript with speakers
  (`txt`) and a per-word timing payload (`words`)
- **Multi-worker Support**: Run multiple workers in parallel for increased throughput

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended package manager)
- FFmpeg (for audio/video processing)
- git, because `nemo-toolkit` is installed from a pinned git commit (see `pyproject.toml`)
- `libsndfile` (`libsndfile1` on Debian), used by NeMo

For GPU use in production (Linux):

- An NVIDIA GPU with a working driver
- `gcc` and the Python development headers (`python3.12-dev` on Debian). NeMo's attention
  layer is compiled by Triton at the first diarization, and the build fails with
  `Python.h: No such file or directory` without them

On a machine without a supported GPU (for example a Mac) everything runs on the CPU. Whisper
falls back to the CPU when the model cannot be loaded on the GPU, and Nemotron only uses CUDA or
the CPU, so transcription is much slower there.

## Development Environment Setup

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd scribe-worker
uv sync
```

### 2. Download the Models

```bash
uv run main.py --download
```

This downloads every Whisper model configured in `utils/settings.py` (`WHISPER_MODELS_HF`),
and the two diarization models (`NEMOTRON_MODEL` and `PYANNOTE_MODEL` in the same file):

- Nemotron 3 Diarization (about 200 MB, no token needed).
- The pyannote pipeline used by the fallback (about 30 MB). It is gated, so it needs `HF_TOKEN`
  with the licence accepted. Without one it is skipped with a message and the rest still
  downloads; a job that needs the fallback will then fail until it is downloaded.

After this the worker can run with `HF_HUB_OFFLINE=1`, and the first job does not wait for a
download.

`download_models.sh` is left over from an earlier whisper.cpp setup and is not used by the
worker.

### 3. Configure Environment Variables

Create a `.env` file in the project root. The settings are defined in `utils/settings.py`; the
ones you usually need are:

```env
# Backend API configuration
API_BACKEND_URL="http://localhost:8000"
API_VERSION="v1"

# Worker configuration
WORKERS=2

# Hugging Face token. Only needed for the pyannote fallback (see below).
HF_TOKEN=<token>

# Diarization models (these are the defaults, see "Speaker diarization")
NEMOTRON_MODEL="nvidia/Nemotron-3-Diarization"
NEMOTRON_MAX_SPEAKERS=8
PYANNOTE_MODEL="pyannote/speaker-diarization-community-1"

# Client certificate for the backend, if it requires one
SSL_CERTFILE=<path>
SSL_KEYFILE=<path>
```

### 4. Run the Worker

```bash
uv run main.py --foreground --debug
```

`uv run main.py --help` lists the other options (`--envfile`, `--logfile`, `--drain` and so on).

## Speaker diarization

The `txt` output carries the transcript split into segments with a speaker for each. Two
engines produce the diarization, and `utils/diarization.py` chooses between them:

| Situation | Engine |
|---|---|
| Up to 8 speakers configured, or none (auto) | Nemotron 3 Diarization |
| More than 8 speakers configured | pyannote (`speaker-diarization-community-1`), with the number as a hint |
| Nemotron reports 8 speakers, which is its limit and could be a cap | pyannote runs as well. Its result is used only if it finds more than 8, otherwise Nemotron's is kept |

The models and the limit are settings (`NEMOTRON_MODEL`, `PYANNOTE_MODEL` and
`NEMOTRON_MAX_SPEAKERS` in `utils/settings.py`, shown above with their defaults), and the
"8" in this section is `NEMOTRON_MAX_SPEAKERS`. `NEMOTRON_MODEL` can be any model that
NeMo's `SortformerEncLabelModel` loads, as long as `NEMOTRON_MAX_SPEAKERS` is the most speakers
that model tells apart. Only Nemotron 3 Diarization has been tested.

Things to know:

- Nemotron always detects the number of speakers itself. The speaker count in a job only
  decides the engine, and pyannote's hint.
- The pyannote model needs `HF_TOKEN`, with the model's licence accepted on huggingface.co.
  Nemotron does not.
- The engine, the number of speakers found and the time taken are logged for every job.

### From diarization to speakers on the text

Whisper gives text with word timings, and the diarization gives who speaks when. The two do not
line up exactly, so `align_speakers()` in `utils/diarization.py` does more than look up the
speaker at the middle of each segment:

- Every word gets a speaker, and a segment is split where the speaker changes.
- Word timings are trimmed to where there is sound in the audio, because Whisper stretches a
  word across the pause next to it.
- Speaker changes in the middle of a sentence are corrected. When one speaker has only a few
  words of a sentence that another speaker says the rest of, the whole sentence goes to the
  main speaker. This covers a turn that starts a moment before the diarization says so, and
  a word that overlaps two speakers.
- A segment does not start before its speaker does, which keeps the timestamps in step with
  the audio when Whisper's text skips speech.

The tolerances are constants at the top of `utils/diarization.py`, with comments on why each has
the value it has. They come from a few real recordings and may need tuning on others.

The `txt` JSON result also includes `diarization_segments`: the raw `{"start", "end", "speaker"}`
intervals returned by whichever diarization engine ran (Nemotron or pyannote), before alignment to
transcript chunks. This preserves overlapping speech, full timestamp precision, and speaker turns
without transcribed words for downstream consumers.

### Privacy

Audio and transcripts stay on the worker. Only the backend API is contacted with them, and the
models are downloaded from Hugging Face. The telemetry of the libraries involved is switched
off in the code (Hugging Face, NeMo and pyannote; pyannote's is on by default and reports the
audio duration). Once the models have been downloaded, `HF_HUB_OFFLINE=1` stops the worker
from contacting Hugging Face at all.

### Troubleshooting

- `Python.h: No such file or directory` at the first diarization: install the Python
  development headers (`python3.12-dev` on Debian).
- `self_attention_model='rope' is not supported`: NeMo is too old for the model. Install with
  `uv sync --frozen`, which uses the pinned NeMo commit, not the release from PyPI.
- `No module named 'torchaudio'` from the voice activity detection: `torchaudio` is a direct
  dependency in `pyproject.toml`, so run `uv sync --frozen` again.

## Docker

Build and run with Docker:

```bash
docker build -t scribe-worker .
docker run --env-file .env scribe-worker
```

The image installs `ffmpeg`, `git`, `gcc` and `libsndfile1`, which the dependencies above need.

## Project Structure

```
scribe-worker/
├── main.py              # Worker entry point
├── utils/
│   ├── args.py          # Command line options
│   ├── settings.py      # Settings (.env)
│   ├── job.py           # One transcription job, from the backend to the result
│   ├── whisper.py       # Transcription, subtitles and the diarization step
│   ├── diarization.py   # Engine choice and speaker alignment
│   ├── nemotron.py      # Nemotron 3 Diarization
│   ├── pyannote.py      # pyannote fallback
│   ├── words.py         # The per-word timing payload
│   ├── media.py         # FFmpeg helpers
│   └── log.py           # Logging
├── models/              # Whisper model files
└── downloaded/          # Downloaded files for processing
```
