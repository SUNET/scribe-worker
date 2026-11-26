FROM python:3.12-slim-bookworm

# Runtime and build dependencies:
#   ffmpeg      audio/video processing
#   git         nemo-toolkit is installed from a pinned git commit
#   gcc         Triton compiles NeMo's attention layer at the first diarization
#   libsndfile1 audio I/O for NeMo
#   curl, ca-certificates  downloads (uv, Hugging Face)
# The Python headers (Python.h, needed together with gcc) are already part of
# the official python image. On a plain Debian/Ubuntu base add python3.12-dev.
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		ca-certificates \
		curl \
		ffmpeg \
		gcc \
		git \
		libsndfile1 && \
	rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Unprivileged user. /data holds everything that should survive a restart:
# the Hugging Face and torch.hub (Silero VAD) model caches and the Triton
# kernel cache.
RUN useradd --create-home --uid 1000 worker && \
	mkdir -p /app /data && \
	chown worker:worker /app /data

USER worker
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
	UV_LINK_MODE=copy \
	UV_NO_DEV=1 \
	PATH="/app/.venv/bin:$PATH" \
	PYTHONUNBUFFERED=1 \
	HF_HOME=/data/huggingface \
	TORCH_HOME=/data/torch \
	TRITON_CACHE_DIR=/data/triton

# Copy dependency files first for better caching
COPY --chown=worker:worker pyproject.toml uv.lock ./

# Install dependencies (torch brings its own CUDA libraries)
RUN uv sync --frozen --no-install-project

# Copy application files
COPY --chown=worker:worker main.py .
COPY --chown=worker:worker utils/ utils/

VOLUME /data

# Download the models once with:
#   docker run --rm --env-file .env -v scribe-data:/data scribe-worker python main.py --download
# The venv is on PATH, so nothing is re-synced when the container starts.
CMD ["python", "main.py", "--foreground", "--no-healthcheck"]
