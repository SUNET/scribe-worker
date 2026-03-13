import io
import os
import subprocess
import tempfile
import wave

from utils import settings

settings = settings.get_settings()


def _run_cmd_pipe(command: list[str]) -> bytes:
    """
    Run a command and return its stdout as bytes.
    Raises an exception with stderr details on failure.
    """
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {result.stderr.decode(errors='replace')}"
        )

    return result.stdout


def has_video_stream(input_path: str) -> bool:
    """
    Check if the input file contains a video stream using ffprobe.
    """
    ffprobe_path = os.path.join(
        os.path.dirname(settings.FFMPEG_PATH), "ffprobe"
    ) if os.path.dirname(settings.FFMPEG_PATH) else "ffprobe"

    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                input_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0 and b"video" in result.stdout
    except Exception:
        return False


def downscale_video(input_path: str) -> bytes:
    """
    Downscale video to a smaller size.
    Returns the MP4 data as bytes.
    """
    command = [
        settings.FFMPEG_PATH,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        input_path,
        "-vf",
        "scale=-2:240:flags=bilinear",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-profile:v",
        "baseline",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "24",
        "-keyint_min",
        "24",
        "-c:a",
        "aac",
        "-b:a",
        "48k",
        "-ar",
        "44100",
        "-ac",
        "1",
        "-movflags",
        "frag_keyframe+empty_moov+default_base_moof",
        "-f",
        "mp4",
        "pipe:1",
    ]

    return _run_cmd_pipe(command)


def downsample_audio(input_path: str) -> bytes:
    """
    Downsample audio to an MP4 with faststart (moov atom at the start).
    Uses a temp file since faststart requires a seekable output.
    Returns the MP4 data as bytes.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        command = [
            settings.FFMPEG_PATH,
            "-nostdin",
            "-threads",
            "0",
            "-i",
            input_path,
            "-c:a",
            "aac",
            "-b:a",
            "48k",
            "-ar",
            "44100",
            "-ac",
            "1",
            "-movflags",
            "+faststart",
            "-y",
            tmp.name,
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed (exit {result.returncode}): "
                f"{result.stderr.decode(errors='replace')}"
            )

        return tmp.read()


def transcode_to_wav(input_path: str) -> bytes:
    """
    Transcode to 16kHz mono signed 16-bit PCM wrapped in a WAV header.
    Returns WAV data as bytes.
    """
    command = [
        settings.FFMPEG_PATH,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "s16le",
        "pipe:1",
    ]

    pcm_data = _run_cmd_pipe(command)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_data)
    return buf.getvalue()
