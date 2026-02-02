# scribe-worker

Worker for the SUNET transcription service (Sunet Scribe).

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
