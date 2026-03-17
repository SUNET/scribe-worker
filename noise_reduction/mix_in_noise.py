import numpy as np
import soundfile as sf
import subprocess
import argparse
from pathlib import Path

def extract_audio(file, out_file):
    cmd = [
        "ffmpeg",
        "-i", file,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        out_file
    ]
    subprocess.run(cmd, check=True)

def reinsert_audio(video_file, audio_file, out_file):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_file),
        "-i", str(audio_file),
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-ar", "16000",
        str(out_file)
    ]
    subprocess.run(cmd, check=True)


def rms(x):
    return np.sqrt(np.mean(x**2))

def mix_at_snr(clean, noise, snr_db):
    clean_rms = rms(clean)
    noise_rms = rms(noise)

    target_noise_rms = clean_rms / (10**(snr_db/20))
    noise_scaled = noise * (target_noise_rms / noise_rms)

    noisy = clean + noise_scaled
    return noisy


def create_noisy_version(video, noises, snrs, output_dir):
    file_name = str(video).split('/')[-1].split('.')[0]
    extracted_audio = f"{output_dir}/{file_name}_temp.wav"
    out_file = f"{output_dir}/{file_name}_noisy.mp4"
    extract_audio(str(video), extracted_audio)
    rng = np.random.default_rng()
    noise_index = rng.integers(low=0, high=len(noises) -1, size=1)[0]
    noise = noises[noise_index]

    snr_index = rng.integers(low=0, high=len(snrs) -1, size=1)[0]
    snr = snrs[snr_index]

    clean_audio, sr = sf.read(extracted_audio)
    noise_audio, _ = sf.read(noise)

    # build noise using random segments instead of repeating the same noise
    rng = np.random.default_rng()

    noise_len = len(noise_audio)
    target_len = len(clean_audio)

    segments = []
    remaining = target_len

    while remaining > 0:
        if noise_len > remaining:
            start = rng.integers(0, noise_len - remaining)
            segment = noise_audio[start:start + remaining]
        else:
            start = rng.integers(0, max(1, noise_len - remaining))
            segment = noise_audio[start:start + min(noise_len, remaining)]

        segments.append(segment)
        remaining -= len(segment)

    noise_audio = np.concatenate(segments)
    noise_audio = noise_audio[:target_len]

    noise_audio = noise_audio[:len(clean_audio)]

    noisy_audio = mix_at_snr(clean_audio, noise_audio, snr)

    noisy_audio_file = f"{output_dir}/{file_name}_noisy.wav"
    sf.write(noisy_audio_file, noisy_audio, sr)

    reinsert_audio(video, noisy_audio_file, out_file)

    return {"file_name": file_name, "snr": snr, "noise": noise}



def main():
    parser = argparse.ArgumentParser(
        description="Generate noisy evaluation dataset from clean videos"
    )
    parser.add_argument("-cvd", "--clean_videos_dir", help="Folder with clean MP4 videos")
    parser.add_argument("-nd", "--noise_dir", help="Folder with noise WAV files")
    parser.add_argument("-out", "--output_dir", help="Output folder")
    parser.add_argument(
        "--snr",
        nargs="+",
        type=float,
        default=[30, 30],
        help="SNR levels (default: 30)",
    )

    args = parser.parse_args()

    clean_dir = Path(args.clean_videos_dir)
    noise_dir = Path(args.noise_dir)
    output_dir = Path(args.output_dir)

    videos = list(clean_dir.glob("*.mp4"))
    noises = list(noise_dir.glob("*.wav"))

    if not videos:
        raise RuntimeError("No MP4 files found in clean_videos_dir")

    if not noises:
        raise RuntimeError("No WAV files found in noise_dir")

    print(f"Found {len(videos)} videos")
    print(f"Found {len(noises)} noise files")
    print(f"SNR levels: {args.snr}")

    for video in videos:
        create_noisy_version(video, noises, args.snr, output_dir)


if __name__ == "__main__":
    main()


