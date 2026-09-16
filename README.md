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

## Raw speaker intervals in JSON results

New diarized JSON results include an additive `diarization_segments` field:

```json
{
  "diarization_segments": [
    {"start": 0.123456789, "end": 2.5, "speaker": "Speaker_00"},
    {"start": 1.7, "end": 3.0, "speaker": "Speaker_01"}
  ]
}
```

These are the original `speaker_diarization` intervals from pyannote, in
seconds relative to the beginning of the recording. Overlapping intervals
remain overlapping. Intervals without transcribed words are retained, and
times are not rounded or adjusted to match transcript segments. Speaker
identifiers use the same naming convention as the transcript.

The existing `segments`, `full_transcription`, and `speaker_count` fields
are unchanged. This field is included in the existing JSON result upload;
it does not introduce an endpoint, another inference pass, or a new result
format. It is not added to SRT or the separate word-timing payload.

An empty list means diarization returned no intervals. A missing field means
the result predates this addition or was rewritten by a consumer that does
not preserve it. Consumers should read it from the original worker result;
it is model output, not a representation of subsequent transcript edits.

Run the model-free serialization tests with
`python -m unittest discover -s tests`.

## Features

- **Transcription Processing**: Processes audio/video transcription jobs from the backend queue
- **Whisper.cpp Integration**: Uses whisper.cpp for efficient local transcription
- **Multiple Output Formats**: Generates JSON and SRT transcription outputs
- **Multi-worker Support**: Run multiple workers in parallel for increased throughput

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended package manager)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (must be built separately)
- FFmpeg (for audio/video processing)

## Development Environment Setup

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd scribe-worker
uv sync
```

### 2. Build whisper.cpp

Build and install whisper.cpp from source. See https://github.com/ggml-org/whisper.cpp for detailed instructions.

### 3. Download Whisper Models

```bash
./download_models.sh
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with the following settings:

```env
# Debug mode
DEBUG=True

# Backend API configuration
API_BACKEND_URL="http://localhost:8000"
API_VERSION="v1"

# Worker configuration
WORKERS=2
WHISPER_CPP_PATH=<Path to whisper.cpp>
FILE_STORAGE_DIR=<Your file storage directory>
```

### 5. Run the Worker

```bash
uv run main.py --foreground --debug
```

## Roles: transcription and inference

A worker does transcription unless told otherwise. `--role` selects what it
handles:

```bash
uv run main.py --foreground --role transcription   # the default
uv run main.py --foreground --role inference       # language model only
uv run main.py --foreground --role transcription,inference
uv run main.py --foreground --role transcription --role inference
```

The last two mean the same thing: repeating the flag adds a role rather
than replacing the one before it.

Production splits the two across hosts. A language model is kept resident on
the GPU -- loading tens of gigabytes of weights takes far longer than
answering with them -- and a Whisper model does not fit beside it.

An inference worker does not poll for work. It connects out to the inference
hub and waits, presenting the same client certificate the REST calls use, and
the hub pushes a request the moment a reader asks for one. There is no queue
to poll because inference requests are never stored: they exist in the hub's
memory and in this process while being answered, and nowhere else.

### Inference configuration

```env
# Where the hub listens. The worker connects out to this.
INFERENCE_HUB_URL="wss://scribe.example.se/api/v1/ws/worker-inference"

# How this worker names itself. Empty means the hostname.
INFERENCE_WORKER_ID=""

# Requests taken at once. One: a second generation on the same card makes
# both slower and risks running out of VRAM mid-answer.
INFERENCE_CONCURRENCY=1

# Refuse to talk to a model server that is not local. Transcripts are the
# reason this service is self-hosted at all.
INFERENCE_ALLOW_REMOTE=False

# Load models at startup rather than on the first request. Right in
# production; on a development box set it False so the worker does not
# download tens of gigabytes before you have tried anything.
INFERENCE_PRELOAD=True

# The model registry. Optional -- utils/settings.py carries the production
# default. Quoted, so it may span lines.
INFERENCE_MODELS='{"gemma": {"provider": "openai",
                             "model_id": "gemma3:12b",
                             "base_url": "http://127.0.0.1:11434/v1"}}'
```

### Models

`INFERENCE_MODELS` maps an alias -- what the frontend shows, and what usage
records name -- to a provider and its settings. The production default is in
`utils/settings.py`; `.env` replaces the whole map. Adding a model to compare
against is a configuration change and a restart, never a code change:

```env
INFERENCE_MODELS='{
  "gemma":      {"provider": "transformers",
                 "model_id": "google/gemma-3-27b-it",
                 "temperature": 0.3},
  "gemma-vllm": {"provider": "openai",
                 "model_id": "google/gemma-3-27b-it",
                 "base_url": "http://127.0.0.1:8000/v1"}
}'
```

The map is checked when the worker starts: an unknown provider, a missing
`model_id`, or an `openai` entry with no `base_url` stops it there, naming
the alias. The alternative is a reader clicking Analyse, waiting for a
worker, and being told the model is unusable long after whoever wrote the
entry has gone home.

Two providers:

- `transformers` loads the model into the worker process and keeps it there.
  `model_id` is a Hugging Face repository; gated ones need `HF_TOKEN`.
- `openai` talks to a local OpenAI-compatible server -- vLLM, llama.cpp's
  server, or Ollama -- through `base_url`. The address is checked before
  anything is sent to it and must resolve to a loopback, private or
  link-local address unless `INFERENCE_ALLOW_REMOTE` is set. A model endpoint
  on the public internet would forward every transcript off-site, which is
  the one thing the architecture exists to prevent.

Prompts are not configured here. They live on the server, versioned, and
arrive with each request -- which is what makes it possible to say later
which prompt produced which answer.

## Docker

Build and run with Docker:

```bash
docker build -t scribe-worker .
docker run --env-file .env scribe-worker
```

## Project Structure

```
scribe-worker/
├── main.py              # Worker entry point
├── utils/               # Utility modules
├── models/              # Whisper model files
└── downloaded/          # Downloaded files for processing
```
