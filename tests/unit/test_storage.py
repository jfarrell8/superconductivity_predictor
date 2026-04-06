"""
tests/unit/test_storage.py
───────────────────────────
Unit tests for the LocalBackend and S3Backend using moto to mock AWS.
No real AWS credentials or network calls are made.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.storage.store import LocalBackend, S3Backend, StorageBackend

# ─── LocalBackend ─────────────────────────────────────────────────────────────


class TestLocalBackend:
    def test_upload_and_download_roundtrip(self, tmp_path: Path) -> None:
        store = LocalBackend(base_dir=tmp_path)
        src = tmp_path / "source.txt"
        src.write_text("hello world")

        store.upload(src, "subdir/dest.txt")
        dest = tmp_path / "download.txt"
        store.download("subdir/dest.txt", dest)
        assert dest.read_text() == "hello world"

    def test_exists_true(self, tmp_path: Path) -> None:
        store = LocalBackend(base_dir=tmp_path)
        (tmp_path / "file.txt").write_text("x")
        assert store.exists("file.txt")

    def test_exists_false(self, tmp_path: Path) -> None:
        store = LocalBackend(base_dir=tmp_path)
        assert not store.exists("nonexistent.txt")

    def test_write_and_read_csv(self, tmp_path: Path) -> None:
        store = LocalBackend(base_dir=tmp_path)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        store.write_csv(df, "data/test.csv", index=False)
        loaded = store.read_csv("data/test.csv")
        pd.testing.assert_frame_equal(df, loaded)

    def test_write_and_read_bytes(self, tmp_path: Path) -> None:
        store = LocalBackend(base_dir=tmp_path)
        data = b"\x00\x01\x02\x03"
        store.write_bytes(data, "binary/blob.bin")
        assert store.read_bytes("binary/blob.bin") == data

    def test_nested_directories_created_automatically(self, tmp_path: Path) -> None:
        store = LocalBackend(base_dir=tmp_path)
        store.write_bytes(b"test", "a/b/c/d/file.bin")
        assert (tmp_path / "a/b/c/d/file.bin").exists()


# ─── S3Backend ────────────────────────────────────────────────────────────────


@pytest.fixture()
def s3_store():
    """Create an S3Backend with a moto-mocked S3 bucket."""
    moto = pytest.importorskip("moto", reason="moto not installed")

    with moto.mock_aws():
        import boto3

        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")

        store = S3Backend(bucket="test-bucket", prefix="test-prefix", region="us-east-1")
        store._client = client  # inject mocked client
        yield store


class TestS3Backend:
    def test_write_and_read_bytes(self, s3_store: S3Backend) -> None:
        data = b"hello s3"
        s3_store.write_bytes(data, "artifacts/file.bin")
        assert s3_store.read_bytes("artifacts/file.bin") == data

    def test_exists_true(self, s3_store: S3Backend) -> None:
        s3_store.write_bytes(b"x", "models/model.pkl")
        assert s3_store.exists("models/model.pkl")

    def test_exists_false(self, s3_store: S3Backend) -> None:
        assert not s3_store.exists("models/nonexistent.pkl")

    def test_write_and_read_csv(self, s3_store: S3Backend) -> None:
        df = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        s3_store.write_csv(df, "data/test.csv", index=False)
        loaded = s3_store.read_csv("data/test.csv")
        pd.testing.assert_frame_equal(df, loaded)

    def test_upload_and_download(self, s3_store: S3Backend, tmp_path: Path) -> None:
        src = tmp_path / "upload.txt"
        src.write_text("uploaded content")
        s3_store.upload(src, "test/upload.txt")

        dest = tmp_path / "downloaded.txt"
        s3_store.download("test/upload.txt", dest)
        assert dest.read_text() == "uploaded content"

    def test_prefix_is_prepended(self, s3_store: S3Backend) -> None:
        key = s3_store._key("models/model.pkl")
        assert key == "test-prefix/models/model.pkl"

    def test_empty_prefix(self) -> None:
        moto = pytest.importorskip("moto")
        with moto.mock_aws():
            import boto3

            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="no-prefix-bucket")
            store = S3Backend(bucket="no-prefix-bucket", prefix="", region="us-east-1")
            store._client = client
            key = store._key("models/model.pkl")
            assert key == "models/model.pkl"


# ─── Factory ──────────────────────────────────────────────────────────────────


class TestStorageBackendFactory:
    def test_local_backend_by_default(self) -> None:
        cfg: dict = {}
        store = StorageBackend.from_config(cfg)
        assert isinstance(store, LocalBackend)

    def test_local_backend_explicit(self) -> None:
        cfg = {"storage": {"backend": "local"}}
        store = StorageBackend.from_config(cfg)
        assert isinstance(store, LocalBackend)

    def test_s3_backend_from_config(self) -> None:
        cfg = {
            "storage": {
                "backend": "s3",
                "s3_bucket": "my-bucket",
                "s3_prefix": "v1",
                "aws_region": "us-west-2",
            }
        }
        store = StorageBackend.from_config(cfg)
        assert isinstance(store, S3Backend)
        assert store.bucket == "my-bucket"
        assert store.prefix == "v1"
        assert store.region == "us-west-2"
