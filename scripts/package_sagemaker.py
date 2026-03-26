"""
scripts/package_sagemaker.py
─────────────────────────────
Packages trained model artifacts into the tar.gz bundle that SageMaker
expects, then optionally uploads it to S3.

SageMaker built-in container requirements
──────────────────────────────────────────
The sklearn container looks for:
  model.tar.gz/
  ├── best_model_top15.pkl   — fitted estimator (joblib)
  ├── top_features.json      — ordered list of feature names
  └── inference.py           — entry-point script (our custom one)

The container mounts this to /opt/ml/model/ and calls model_fn().

Usage
─────
    # Package only (inspect before uploading)
    python scripts/package_sagemaker.py

    # Package and upload to S3
    python scripts/package_sagemaker.py --upload

    # Override paths
    python scripts/package_sagemaker.py \\
        --model-dir models/ \\
        --output models/best_model_top15.tar.gz \\
        --upload \\
        --bucket my-bucket \\
        --prefix superconductivity_v1/models
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


REQUIRED_FILES = {
    "best_model_top15.pkl": "Fitted model (joblib)",
    "top_features.json": "Top-15 feature names",
}
INFERENCE_SCRIPT = Path("src/api/sagemaker/inference.py")


def package(
    model_dir: Path,
    output_path: Path,
) -> Path:
    """
    Bundle model artifacts and inference script into a tar.gz.

    Parameters
    ----------
    model_dir   : directory containing model artifacts
    output_path : where to write the .tar.gz

    Returns
    -------
    Path to the created archive.
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    missing = [f for f in REQUIRED_FILES if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Required artifacts not found in {model_dir}: {missing}\n"
            "Run `make train` or `make flow` first."
        )
    if not INFERENCE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Inference script not found at {INFERENCE_SCRIPT}. "
            "Is the project root your working directory?"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Build archive ─────────────────────────────────────────────────────────
    with tarfile.open(output_path, "w:gz") as tar:
        for filename in REQUIRED_FILES:
            src = model_dir / filename
            tar.add(src, arcname=filename)
            logger.info(f"  + {filename}  ({src.stat().st_size:,} bytes)")

        tar.add(INFERENCE_SCRIPT, arcname="inference.py")
        logger.info(f"  + inference.py")

    size_kb = output_path.stat().st_size / 1024
    logger.info(f"Archive created: {output_path}  ({size_kb:.1f} KB)")
    return output_path


def upload_to_s3(
    archive_path: Path,
    bucket: str,
    prefix: str,
    region: str = "us-east-1",
) -> str:
    """
    Upload the tar.gz to S3 and return the S3 URI.
    Uses boto3 — credentials resolved from the standard chain (env > ~/.aws > IAM role).
    """
    import boto3

    s3_key = f"{prefix.rstrip('/')}/{archive_path.name}"
    s3_uri = f"s3://{bucket}/{s3_key}"

    logger.info(f"Uploading {archive_path} → {s3_uri}")
    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(str(archive_path), bucket, s3_key)
    logger.info(f"Upload complete: {s3_uri}")
    return s3_uri


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package model artifacts for SageMaker deployment."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained model artifacts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/best_model_top15.tar.gz",
        help="Output path for the tar.gz bundle.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the bundle to S3 after packaging.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="S3 bucket name (required with --upload).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="models",
        help="S3 key prefix (default: models).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region (default: us-east-1).",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_path = Path(args.output)

    logger.info(f"Packaging SageMaker bundle from {model_dir}/")
    archive = package(model_dir, output_path)

    if args.upload:
        if not args.bucket:
            # Fall back to config if available
            try:
                import yaml
                cfg = yaml.safe_load(open("configs/experiment_v1.yaml"))
                bucket = cfg["storage"]["s3_bucket"]
                logger.info(f"Using bucket from config: {bucket}")
            except Exception:
                logger.error("--bucket is required when using --upload.")
                sys.exit(1)
        else:
            bucket = args.bucket

        s3_uri = upload_to_s3(archive, bucket, args.prefix, args.region)
        print(f"\nS3 URI (use in Terraform or SageMaker console):\n  {s3_uri}")
    else:
        print(f"\nBundle ready: {archive}")
        print("Run with --upload to push to S3, or:")
        print(f"  aws s3 cp {archive} s3://YOUR_BUCKET/models/")


if __name__ == "__main__":
    main()
