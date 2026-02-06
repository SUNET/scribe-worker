FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

# Build CUDA-accelerated ffmpeg
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		build-essential \
		git \
		nasm \
		yasm \
		pkg-config \
		libx264-dev \
		libx265-dev \
		libnuma-dev \
		libvpx-dev \
		libfdk-aac-dev \
		libmp3lame-dev \
		libopus-dev && \
	rm -rf /var/lib/apt/lists/*

# Clone and build nv-codec-headers
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git /tmp/nv-codec-headers && \
	cd /tmp/nv-codec-headers && \
	make install

# Clone and build ffmpeg with CUDA support
RUN git clone https://git.ffmpeg.org/ffmpeg.git /tmp/ffmpeg && \
	cd /tmp/ffmpeg && \
	./configure \
		--enable-gpl \
		--enable-nonfree \
		--enable-cuda-nvcc \
		--enable-libnpp \
		--enable-cuvid \
		--enable-nvenc \
		--enable-nvdec \
		--enable-libx264 \
		--enable-libx265 \
		--enable-libvpx \
		--enable-libfdk-aac \
		--enable-libmp3lame \
		--enable-libopus \
		--extra-cflags="-I/usr/local/cuda/include" \
		--extra-ldflags="-L/usr/local/cuda/lib64" && \
	make -j$(nproc) && \
	make install

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Copy ffmpeg from builder
COPY --from=builder /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=builder /usr/local/bin/ffprobe /usr/local/bin/ffprobe
COPY --from=builder /usr/local/lib/libav* /usr/local/lib/
COPY --from=builder /usr/local/lib/libsw* /usr/local/lib/
COPY --from=builder /usr/local/lib/libpostproc* /usr/local/lib/

# Install runtime dependencies
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		ca-certificates \
		curl \
		git \
		libx264-163 \
		libx265-199 \
		libvpx7 \
		libfdk-aac2 \
		libmp3lame0 \
		libopus0 \
		libnuma1 \
		python3 \
		python3-dev \
		python3-pip && \
	rm -rf /var/lib/apt/lists/*

# Update library cache
RUN ldconfig

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy application files
COPY pyproject.toml uv.lock ./
COPY main.py .
COPY utils/ utils/

# Install Python dependencies with uv
RUN uv sync --frozen

# Run worker
CMD ["uv", "run", "python", "main.py", "--foreground", "--debug"]
