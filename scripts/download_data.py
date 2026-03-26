"""
scripts/download_data.py
─────────────────────────
Downloads the UCI Superconductivity dataset and places it in data/raw/.

Dataset: https://archive.ics.uci.edu/dataset/464/superconductivty+data
  - train.csv    : 21,263 samples × 82 columns (81 features + critical_temp)
  - unique_m.csv : chemical formula metadata

Usage
─────
    python scripts/download_data.py
    python scripts/download_data.py --output-dir custom/path
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

from loguru import logger

# Mirror URL — UCI dataset via a stable public link
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
)
EXPECTED_FILES = ["train.csv", "unique_m.csv"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "superconductivity.zip"

    # Check if already downloaded
    if all((output_dir / f).exists() for f in EXPECTED_FILES):
        logger.info("Data files already present — skipping download.")
        return

    logger.info(f"Downloading dataset from {DATASET_URL} …")
    try:
        urllib.request.urlretrieve(DATASET_URL, zip_path)
    except Exception as exc:
        logger.error(
            f"Download failed: {exc}\n"
            "Please download manually from:\n"
            "  https://archive.ics.uci.edu/dataset/464/superconductivty+data\n"
            f"and place train.csv and unique_m.csv in {output_dir}/"
        )
        sys.exit(1)

    logger.info(f"Extracting to {output_dir}/")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    zip_path.unlink()  # remove the zip after extraction

    for fname in EXPECTED_FILES:
        fpath = output_dir / fname
        if fpath.exists():
            sha = _sha256(fpath)
            logger.info(f"  {fname}  sha256={sha[:16]}…")
        else:
            logger.warning(f"Expected file not found after extraction: {fname}")

    logger.info("Download complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the UCI superconductivity dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save raw data files (default: data/raw).",
    )
    args = parser.parse_args()
    download(Path(args.output_dir))


if __name__ == "__main__":
    main()
