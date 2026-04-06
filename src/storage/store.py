"""
src/storage/store.py
─────────────────────
Unified artifact and data storage layer with swappable backends.

The rest of the codebase never imports boto3 or touches S3 paths directly —
it always goes through a StorageBackend. This means:
  • Local development: zero AWS credentials required
  • CI/CD: mock with moto
  • Production: pass backend="s3" and everything else is unchanged

Backends
────────
LocalBackend  — reads/writes the local filesystem (default, zero config)
S3Backend     — reads/writes AWS S3 via boto3; handles multipart uploads,
                presigned URLs, and existence checks

Usage
─────
    # Anywhere in the pipeline
    store = StorageBackend.from_config(cfg)

    store.upload("data/processed/train_reduced.csv", "data/processed/train_reduced.csv")
    store.download("models/best_model.pkl", "/tmp/best_model.pkl")
    df = store.read_csv("data/raw/train.csv")
    df.to_csv(store.open_write("data/processed/out.csv"), index=False)
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

# ─── Abstract base ────────────────────────────────────────────────────────────


class StorageBackend(ABC):
    """Protocol that all storage backends must implement."""

    @abstractmethod
    def upload(self, local_path: str | Path, remote_key: str) -> None:
        """Copy a local file to remote storage."""

    @abstractmethod
    def download(self, remote_key: str, local_path: str | Path) -> None:
        """Fetch a remote file to local disk."""

    @abstractmethod
    def exists(self, remote_key: str) -> bool:
        """Return True if the remote key exists."""

    @abstractmethod
    def read_csv(self, remote_key: str, **kwargs: Any) -> pd.DataFrame:
        """Read a CSV directly from remote storage into a DataFrame."""

    @abstractmethod
    def write_csv(self, df: pd.DataFrame, remote_key: str, **kwargs: Any) -> None:
        """Write a DataFrame as CSV directly to remote storage."""

    @abstractmethod
    def read_bytes(self, remote_key: str) -> bytes:
        """Return the raw bytes of a remote object."""

    @abstractmethod
    def write_bytes(self, data: bytes, remote_key: str) -> None:
        """Write raw bytes to a remote object."""

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> StorageBackend:
        """Instantiate the correct backend from a parsed YAML config."""
        storage_cfg = cfg.get("storage", {})
        backend = storage_cfg.get("backend", "local")
        if backend == "s3":
            return S3Backend(
                bucket=storage_cfg["s3_bucket"],
                prefix=storage_cfg.get("s3_prefix", ""),
                region=storage_cfg.get("aws_region", "us-east-1"),
            )
        return LocalBackend(base_dir=Path("."))


# ─── Local backend ────────────────────────────────────────────────────────────


class LocalBackend(StorageBackend):
    """
    Filesystem-backed storage. Remote keys are treated as paths relative
    to base_dir. Ideal for local development and CI.
    """

    def __init__(self, base_dir: Path | str = Path(".")) -> None:
        self.base_dir = Path(base_dir)

    def _resolve(self, key: str) -> Path:
        path = self.base_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def upload(self, local_path: str | Path, remote_key: str) -> None:
        import shutil

        dest = self._resolve(remote_key)
        shutil.copy2(str(local_path), str(dest))
        logger.debug(f"[local] upload {local_path} → {dest}")

    def download(self, remote_key: str, local_path: str | Path) -> None:
        import shutil

        src = self._resolve(remote_key)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(local_path))
        logger.debug(f"[local] download {src} → {local_path}")

    def exists(self, remote_key: str) -> bool:
        return (self.base_dir / remote_key).exists()

    def read_csv(self, remote_key: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(self._resolve(remote_key), **kwargs)

    def write_csv(self, df: pd.DataFrame, remote_key: str, **kwargs: Any) -> None:
        df.to_csv(self._resolve(remote_key), **kwargs)
        logger.debug(f"[local] wrote CSV to {self._resolve(remote_key)}")

    def read_bytes(self, remote_key: str) -> bytes:
        return (self.base_dir / remote_key).read_bytes()

    def write_bytes(self, data: bytes, remote_key: str) -> None:
        self._resolve(remote_key).write_bytes(data)


# ─── S3 backend ───────────────────────────────────────────────────────────────


class S3Backend(StorageBackend):
    """
    AWS S3-backed storage.

    All keys are namespaced under `{bucket}/{prefix}/`.
    Credentials are resolved by boto3 in standard order:
      1. Environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
      2. ~/.aws/credentials
      3. IAM instance role (EC2 / ECS / SageMaker)

    No credentials are ever hardcoded or passed as arguments — this is
    intentional. In production, the ECS task / SageMaker training job
    assumes an IAM role with the required S3 permissions.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.region = region
        self._client: Any = None  # lazy init — avoids boto3 import at module level

    def _s3(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def _key(self, remote_key: str) -> str:
        """Prepend the prefix to a relative key."""
        if self.prefix:
            return f"{self.prefix}/{remote_key.lstrip('/')}"
        return remote_key.lstrip("/")

    def upload(self, local_path: str | Path, remote_key: str) -> None:
        full_key = self._key(remote_key)
        logger.info(f"[s3] uploading {local_path} → s3://{self.bucket}/{full_key}")
        self._s3().upload_file(str(local_path), self.bucket, full_key)

    def download(self, remote_key: str, local_path: str | Path) -> None:
        full_key = self._key(remote_key)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"[s3] downloading s3://{self.bucket}/{full_key} → {local_path}")
        self._s3().download_file(self.bucket, full_key, str(local_path))

    def exists(self, remote_key: str) -> bool:
        import botocore.exceptions

        try:
            self._s3().head_object(Bucket=self.bucket, Key=self._key(remote_key))
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def read_csv(self, remote_key: str, **kwargs: Any) -> pd.DataFrame:
        full_key = self._key(remote_key)
        logger.debug(f"[s3] read_csv s3://{self.bucket}/{full_key}")
        obj = self._s3().get_object(Bucket=self.bucket, Key=full_key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()), **kwargs)

    def write_csv(self, df: pd.DataFrame, remote_key: str, **kwargs: Any) -> None:
        buf = io.StringIO()
        df.to_csv(buf, **kwargs)
        self.write_bytes(buf.getvalue().encode(), remote_key)
        logger.debug(f"[s3] wrote CSV to s3://{self.bucket}/{self._key(remote_key)}")

    def read_bytes(self, remote_key: str) -> bytes:
        obj = self._s3().get_object(Bucket=self.bucket, Key=self._key(remote_key))
        return obj["Body"].read()  # type: ignore[no-any-return]

    def write_bytes(self, data: bytes, remote_key: str) -> None:
        full_key = self._key(remote_key)
        self._s3().put_object(Bucket=self.bucket, Key=full_key, Body=data)

    def presigned_url(self, remote_key: str, expiry_seconds: int = 3600) -> str:
        """Generate a presigned GET URL for sharing an artifact."""
        return self._s3().generate_presigned_url(  # type: ignore[no-any-return]
            "get_object",
            Params={"Bucket": self.bucket, "Key": self._key(remote_key)},
            ExpiresIn=expiry_seconds,
        )
