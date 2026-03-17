import os
from jiwer import wer, mer, wil
from pathlib import Path
import argparse


def read_transcript(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def collect_videos(clean_dir):
    return sorted([f.stem for f in clean_dir.glob("*.txt")])


def compute_scores(base_dir, model_dirs, method):
    clean_dir = base_dir / "clean"
    noisy_dir = base_dir / "noisy"

    videos = collect_videos(clean_dir)

    results = []

    for vid in videos:
        clean_path = clean_dir / f"{vid}.txt"
        noisy_path = noisy_dir / f"{vid}.txt"

        if not clean_path.exists():
            continue

        ref = read_transcript(clean_path)

        row = {
            "video": vid,
            "noisy": None,
        }

        if noisy_path.exists():
            hyp = read_transcript(noisy_path)
            match method:
                case "WER":
                    row["noisy"] = wer(ref, hyp)
                case "WIL":
                    row["noisy"] = wil(ref, hyp)
                case "MER":
                    row["noisy"] = mer(ref, hyp)
                case _:
                    raise RuntimeError(f"Bad method {method}")

        for model_dir in model_dirs:
            model_name = model_dir.name
            model_path = model_dir / f"{vid}.txt"

            if model_path.exists():
                hyp = read_transcript(model_path)
                match method:
                    case "WER":
                        row[model_name] = wer(ref, hyp)
                    case "WIL":
                        row[model_name] = wil(ref, hyp)
                    case "MER":
                        row[model_name] = mer(ref, hyp)
                    case _:
                        raise RuntimeError(f"Bad method {method}")
            else:
                row[model_name] = None

        results.append(row)

    return results


def print_csv(results, model_dirs):

    model_names = [d.name for d in model_dirs]

    headers = ["video", "noisy"] + model_names
    print(",".join(headers))

    totals = {h: 0 for h in headers if h != "video"}
    counts = {h: 0 for h in headers if h != "video"}

    for r in results:

        row = [r["video"]]

        for h in headers[1:]:
            val = r.get(h)

            if val is None:
                row.append("")
            else:
                row.append(f"{val:.4f}")
                totals[h] += val
                counts[h] += 1

        print(",".join(row))

    # Print averages row
    avg_row = ["AVERAGE"]

    for h in headers[1:]:
        if counts[h] == 0:
            avg_row.append("")
        else:
            avg = totals[h] / counts[h]
            avg_row.append(f"{avg:.4f}")

    print(",".join(avg_row))


def main():
    parser = argparse.ArgumentParser(
        description="Compare sets of transcriptions to determine WER."
    )
    parser.add_argument("dir", help="Transcription directory.")
    parser.add_argument("--method", "-m", help="WER, MER or WIL", default="WER")

    args = parser.parse_args()

    base_dir = Path(args.dir)

    model_dirs = [
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name not in ["clean", "noisy"]
    ]

    results = compute_scores(base_dir, model_dirs, args.method)

    print_csv(results, model_dirs)


if __name__ == "__main__":
    main()