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

The Debian/Ubuntu packages and how to install them are listed under [Docker](#packages-in-the-image).

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
the Silero VAD model, and the two diarization models (`NEMOTRON_MODEL` and `PYANNOTE_MODEL` in
the same file):

- Silero VAD (about 2 MB), which Whisper uses to find speech. It is not on Hugging Face:
  `torch.hub` downloads it from GitHub (`snakers4/silero-vad`, the `master` branch) into
  `$TORCH_HOME/hub`, by default `~/.cache/torch/hub`.
- Nemotron 3 Diarization (about 200 MB, no token needed).
- The pyannote pipeline used by the fallback (about 30 MB). It is gated, so it needs `HF_TOKEN`
  with the licence accepted. Without one it is skipped with a message and the rest still
  downloads; a job that needs the fallback will then fail until it is downloaded.

After this the worker can run with `HF_HUB_OFFLINE=1`, and the first job does not wait for a
download. Silero VAD is then loaded from `$TORCH_HOME` without contacting GitHub.

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

### Privacy

Audio and transcripts stay on the worker. Only the backend API is contacted with them, and the
models are downloaded from Hugging Face, and the Silero VAD model from GitHub. The telemetry of the libraries involved is switched
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

The image is based on `python:3.12-slim-bookworm`, runs as an unprivileged user and installs
the dependencies with `uv sync --frozen`.

### Build

```bash
docker build -t scribe-worker .
```

### Download the models

Models go to `/data` in the container (`HF_HOME=/data/huggingface` for the Hugging Face models
and `TORCH_HOME=/data/torch` for Silero VAD), so keep it in a volume.
Download them once, which needs `HF_TOKEN` in `.env` for the pyannote fallback:

```bash
docker run --rm --env-file .env -v scribe-data:/data scribe-worker python main.py --download
```

### Run

```bash
docker run -d --name scribe-worker --restart unless-stopped \
    --gpus all --env-file .env -v scribe-data:/data scribe-worker
```

- `--gpus all` needs an NVIDIA driver on the host and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Without it the worker runs on the CPU, which is much slower. The CUDA libraries come with the
  PyTorch wheels, so the image needs no CUDA base image.
- `API_BACKEND_URL` in `.env` must be reachable from inside the container. `localhost` is the
  container itself, so use the host name of the backend (or `host.docker.internal`).
- If the backend needs a client certificate, mount it and point `SSL_CERTFILE` and
  `SSL_KEYFILE` at the mounted paths, for example `-v ./certs:/certs:ro`.
- After the models are downloaded, add `-e HF_HUB_OFFLINE=1` to stop all contact with
  Hugging Face.

### Environment variables

The settings in `utils/settings.py` can be given as real environment variables, which take
priority over a `.env` file. `.env` is not in the image (see `.dockerignore`), so pass the
settings when the container starts:

- `--env-file .env` (as above), or `-e NAME=value` for single values, for example
  `-e WORKERS=1`.
- `docker run --env-file` does **not** strip quotes, so `API_BACKEND_URL="http://backend:8000"`
  would keep the quotes in the value. Leave them out in a file used this way:

  ```env
  API_BACKEND_URL=http://backend:8000
  WORKERS=2
  HF_TOKEN=<token>
  ```

- Alternatively mount the file, which the worker reads itself, so quotes work:
  `-v ./.env:/app/.env:ro`.
- Paths such as `SSL_CERTFILE` and `SSL_KEYFILE` are paths inside the container, so mount the
  files and use the mounted paths.
- `HF_TOKEN` shows up in `docker inspect` when given as a variable. Use a mounted `.env` or
  Docker secrets if that matters.

### Docker Compose

`compose.yaml` does the same as the `docker run` commands above, with the GPU, the `/data`
volume and `.env` configured (Compose handles quoted values in `env_file`):

```bash
docker compose build
docker compose run --rm scribe-worker python main.py --download   # once
docker compose up -d
docker compose logs -f
```

### Packages in the image

The same packages are needed to run the worker outside Docker, on Debian or Ubuntu:

| Package | Why |
|---|---|
| `ffmpeg` | Audio and video processing |
| `git` | `nemo-toolkit` is installed from a pinned git commit |
| `gcc` | Triton compiles NeMo's attention layer at the first diarization |
| `python3.12-dev` | `Python.h`, needed together with `gcc`. Already in the official `python` image, so the Dockerfile does not install it |
| `libsndfile1` | Audio file I/O for NeMo |
| `curl`, `ca-certificates` | Downloads |

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg gcc git libsndfile1 python3.12-dev
```

`python3.12-dev` is only in the default repositories of Ubuntu 24.04 and Debian 13. On older
releases use the `deadsnakes` PPA (Ubuntu) or a Python 3.12 build from `uv python install 3.12`
(which already includes the headers).

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
