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

This field is included in the existing JSON result upload;
it does not introduce an endpoint, another inference pass, or a new result
format. It is not added to SRT or the separate word-timing payload.

An empty list means diarization returned no intervals. A missing field means
the result predates this addition or was rewritten by a consumer that does
not preserve it. Consumers should read it from the original worker result;
it is model output, not a representation of subsequent transcript edits.

Run the tests without model downloads or inference with
`python -m unittest discover -s tests`.

## Speaker alignment

For diarized JSON results, the worker uses word timings to split transcription
chunks when the assigned speaker changes. It partitions the original pyannote
intervals into spans with a constant set of simultaneous speakers, then assigns
each word to the state with the greatest temporal overlap. Consecutive words
with the same state remain together, within the existing chunk boundaries.

`speaker` is the sole speaker when the assigned state contains one speaker.
For overlapping speech it is `Unknown`, and `active_speakers` lists the
simultaneous speakers. Equal-duration ties and words mostly in silence are
also `Unknown`, with an empty active-speaker list. Consecutive speakers are
never listed together as though they spoke simultaneously.

Splitting requires complete agreement between word text and chunk text,
ignoring whitespace, and usable word timestamps. Otherwise the original
text is retained as one chunk: a single consistent speech state can label
it, but ambiguous chunks are marked `Unknown`. Punctuation is preserved.
There is no fabricated timing for missing words and no default to Speaker_00.

Word timing and diarization remain model estimates. A word crossing a speaker
boundary is indivisible and is assigned by overlap; this cannot guarantee its
true speaker or separate the words of simultaneous speakers. The unchanged
`diarization_segments` field retains the original boundaries for downstream
analysis, including activity with no transcribed words.

Full transcription text, the separate word payload, and SRT generation are
unchanged. JSON transcript segment boundaries, counts, confidence averages,
and speaker labels can change. Existing saved results are not reprocessed.
The current UI can display `Unknown`, but does not display `active_speakers`
and may merge adjacent captions with the same primary speaker on import.
Showing overlapping-speaker identities in that editor is a separate change.

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
