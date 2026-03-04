FROM python:3.12-slim-bookworm

# Install runtime dependencies
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		ca-certificates \
		curl \
		ffmpeg \
		git && \
	rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application files
COPY main.py .
COPY utils/ utils/

# Run worker
CMD ["uv", "run", "python", "main.py", "--foreground", "--debug", "--no-healthcheck"]
