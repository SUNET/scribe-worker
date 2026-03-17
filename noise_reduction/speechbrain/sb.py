#!/usr/bin/env python3

import argparse
import subprocess
import tempfile
from pathlib import Path

import torchaudio
import torch
from speechbrain.pretrained import SpectralMaskEnhancement


def run_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(result.stderr.decode())
        raise RuntimeError("FFmpeg command failed")


def extract_audio(input_video, output_wav):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(output_wav),
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
        str(output_video),
    ]
    run_cmd(cmd)


def process(input_path, output_dir, enhancer):

    suffix = input_path.suffix.lower()
    if suffix in [".mp4", ".mov", ".mkv", ".avi"]:
        print("Video file detected. Extracting audio...")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_wav = Path(tmpdir) / "extracted.wav"
            enhanced_wav = Path(tmpdir) / "enhanced.wav"

            extract_audio(input_path, tmp_wav)

            print("Running enhancement...")
            enhanced = enhancer.enhance_file(str(tmp_wav))

            torchaudio.save(
                str(enhanced_wav),
                enhanced.unsqueeze(0),
                16000,
            )

            output_video = output_dir / input_path.name

            print("Replacing audio in video...")
            replace_audio(input_path, enhanced_wav, output_video)

            print(f"Done. Output saved to: {output_video}")

    else:
        print("Audio file detected. Enhancing directly...")
        enhanced = enhancer.enhance_file(str(input_path))

        output_path = output_dir / input_path.name

        torchaudio.save(
            str(output_path),
            enhanced.unsqueeze(0),
            16000,
        )

        print(f"Denoised file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Denoise audio using SpeechBrain MetricGAN+ (supports WAV and MP4)"
    )
    parser.add_argument("input_dir", help="Path to input WAV or MP4 file dir")
    parser.add_argument("output_dir", help="Directory where output file will be saved")

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SpeechBrain MetricGAN+ model...")

    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="metricgan-plus-voicebank",
    )

    videos = list(input_path.glob("*.mp4"))

    for video in videos:
        process(Path(video), output_dir, enhancer)



if __name__ == "__main__":
    main()