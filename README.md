# transcribe-worker

Worker service for the SUNET transcription service (Sunet Scribe).

## Features

- **Audio/Video Transcription**: Process transcription jobs using whisper.cpp or HuggingFace models
- **Speaker Diarization**: Identify and label different speakers using pyannote-audio
- **Multi-Language Support**: Support for Swedish, English, Finnish, Danish, Norwegian, and more
- **GPU Acceleration**: Optional NVIDIA GPU support for faster processing
- **Health Monitoring**: Reports system metrics (CPU, memory, GPU) to the backend
- **Daemon Mode**: Run as a background service or in foreground mode

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended package manager)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (for CPU-based transcription)
- ffmpeg (for audio/video processing)
- NVIDIA GPU (optional, for GPU acceleration)

## Development Environment Setup

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd scribe-worker
uv sync
```

### 2. Build whisper.cpp

Build and install whisper.cpp from source. See https://github.com/ggml-org/whisper.cpp for details.

### 3. Download Whisper Models

```bash
./download_models.sh
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with the following settings:

```env
# General configuration
DEBUG=True
WORKERS=2
FILE_STORAGE_DIR=<Your file storage directory>

# API configuration
API_BACKEND_URL="http://localhost:8000"
API_VERSION="v1"

# whisper.cpp configuration
WHISPER_CPP_PATH=<Path to whisper-cli binary>
WHISPER_MODELS_CPP_FILE=<Optional path to JSON file with model mappings>

# Ollama configuration (for inference)
OLLAMA_URL="http://localhost:11434"

# HuggingFace configuration (optional, for HF-based transcription)
HF_WHISPER=False
HF_TOKEN=<Your HuggingFace token>

# SSL configuration (for backend communication)
SSL_CERTFILE=<Path to SSL certificate>
SSL_KEYFILE=<Path to SSL key>
```

### 5. Run the Application

Run in foreground mode for development:
=======
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
>>>>>>> main

```bash
uv run main.py --foreground --debug
```

Run as a daemon:

```bash
uv run main.py
```

Stop the daemon:

```bash
uv run main.py --zap
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--foreground` | Run in foreground mode (don't daemonize) |
| `--debug` | Enable debug logging |
| `--pidfile <path>` | Specify PID file location |
| `--env <path>` | Specify environment file path |
| `--zap` | Stop a running daemon |
| `--no-healthcheck` | Disable health check reporting |


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

## License

See [LICENSE](LICENSE) for details.
