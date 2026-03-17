import subprocess
import tempfile
import os
import sys
from pathlib import Path
import soundfile as sf
import torch

# DeepFilterNet
from df.enhance import enhance, init_df


def run_cmd(cmd):
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        print(process.stderr.decode())
        raise RuntimeError("Command failed")
    return process


def extract_audio(input_video, output_wav):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vn",
        "-ac", "1",
        "-ar", "48000",
        "-f", "wav",
        str(output_wav)
    ]
    run_cmd(cmd)


def replace_audio(original_video, new_audio, output_video):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(original_video),
        "-i", str(new_audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_video)
    ]
    run_cmd(cmd)


def deepfilter_denoise(input_wav, output_wav):
    model, df_state, _ = init_df()

    audio, sr = sf.read(input_wav)

    audio = audio.astype("float32")

    if audio.ndim == 1:
        audio = audio[None, :]

    audio = torch.from_numpy(audio)

    with torch.no_grad():
        enhanced = enhance(model, df_state, audio)

    enhanced = enhanced.squeeze(0).cpu().numpy()

    sf.write(output_wav, enhanced, sr)


def main(input_file, output_dir):

    input_path = Path(input_file)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list(input_path.glob("*.mp4"))

    for video in videos:

        output_video = output_dir / Path(video).name

        with tempfile.TemporaryDirectory() as tmpdir:
            extracted_wav = Path(tmpdir) / "input.wav"
            enhanced_wav = Path(tmpdir) / "enhanced.wav"

            print("Extracting audio...")
            extract_audio(Path(video), extracted_wav)

            print("Running DeepFilterNet...")
            deepfilter_denoise(extracted_wav, enhanced_wav)

            print("Recombining audio with video...")
            replace_audio(Path(video), enhanced_wav, output_video)

            print("Done:", output_video)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python denoise_video.py input.mp4 output_dir")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])