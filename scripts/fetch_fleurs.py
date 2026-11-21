# Copyright (c) 2025-2026 Sunet.
# Contributor: Kristofer Hallin
#
# This file is part of Sunet Scribe.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Download clean clips with reference text from FLEURS (CC-BY-4.0), as the input
for build_noisy_set.py and eval.py.

    uv run python scripts/fetch_fleurs.py OUT_DIR --language sv_se --split dev --count 40

Writes OUT_DIR/clips/*.wav (16 kHz mono) and OUT_DIR/clean.tsv with one
"path<TAB>reference" line per clip. The reference is FLEURS' normalised
transcription: lower case, no punctuation. The archive (about 190 MB for the
Swedish dev split, 440 MB for test) is cached by huggingface_hub, and the
clips are a seeded random sample, so speakers and sentences vary.
"""

import argparse
import random
import subprocess
import tarfile
from pathlib import Path

from evallib import log


def fetch(out_dir: Path, language: str, split: str, count: int, seed: int = 1) -> Path:
    """Write `count` clips of a FLEURS split to out_dir. Returns clean.tsv."""
    from huggingface_hub import hf_hub_download

    manifest = out_dir / "clean.tsv"

    if manifest.exists() and len(manifest.read_text(encoding="utf-8").splitlines()) >= count:
        log(f"{manifest} already has {count} clips")
        return manifest

    log(f"downloading FLEURS {language} {split} (cached after the first time) ...")
    tsv = hf_hub_download("google/fleurs", f"data/{language}/{split}.tsv", repo_type="dataset")
    archive = hf_hub_download("google/fleurs", f"data/{language}/audio/{split}.tar.gz", repo_type="dataset")

    # Columns: id, file name, raw text, normalised text, tokenised, samples, gender
    rows = [line.split("\t") for line in Path(tsv).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = random.Random(seed).sample(rows, min(count, len(rows)))
    wanted = {row[1]: row[3].strip() for row in rows}

    (out_dir / "clips").mkdir(parents=True, exist_ok=True)
    lines = []

    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = Path(member.name).name

            if name not in wanted or not member.isfile():
                continue

            target = out_dir / "clips" / f"{language}_{split}_{len(lines):03d}.wav"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", "-", "-ac", "1", "-ar", "16000", str(target)],
                input=tar.extractfile(member).read(),
                check=True,
            )
            lines.append(f"{target.resolve()}\t{wanted[name]}")

            if len(lines) == len(wanted):
                break

    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"{len(lines)} clips in {manifest}")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("out_dir")
    parser.add_argument("--language", default="sv_se", help="FLEURS config, e.g. sv_se")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    fetch(Path(args.out_dir), args.language, args.split, args.count, args.seed)


if __name__ == "__main__":
    main()
