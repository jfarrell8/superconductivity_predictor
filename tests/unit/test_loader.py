"""
tests/unit/test_loader.py
──────────────────────────
Unit tests for src/data/loader.py — filesystem operations are faked
using pytest's tmp_path fixture so no real data files are needed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import DataLoader, DataManifest

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _write_valid_csv(path, n_cols: int = 82) -> None:
    """Write a minimal valid superconductivity-like CSV to path."""
    cols = ["critical_temp", "number_of_elements"] + [f"feat_{i}" for i in range(n_cols - 2)]
    df = pd.DataFrame([[float(j) for j in range(n_cols)]] * 10, columns=cols)
    df.to_csv(path, index=False)


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestDataLoader:
    def test_load_returns_self(self, tmp_path) -> None:
        csv = tmp_path / "train.csv"
        _write_valid_csv(csv)
        loader = DataLoader(train_path=csv)
        result = loader.load()
        assert result is loader, ".load() should return self for fluent chaining."

    def test_train_property_returns_copy(self, tmp_path) -> None:
        csv = tmp_path / "train.csv"
        _write_valid_csv(csv)
        loader = DataLoader(train_path=csv).load()
        df1 = loader.train
        df1["__sentinel__"] = 99
        df2 = loader.train
        assert "__sentinel__" not in df2.columns, ".train should return a copy."

    def test_raises_on_missing_file(self, tmp_path) -> None:
        loader = DataLoader(train_path=tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_raises_without_required_column(self, tmp_path) -> None:
        csv = tmp_path / "bad.csv"
        df = pd.DataFrame({"some_col": [1.0, 2.0], "other_col": [3.0, 4.0]})
        df.to_csv(csv, index=False)
        loader = DataLoader(train_path=csv)
        with pytest.raises(ValueError, match="missing required columns"):
            loader.load()

    def test_raises_on_too_few_columns(self, tmp_path) -> None:
        csv = tmp_path / "sparse.csv"
        df = pd.DataFrame(
            {
                "critical_temp": [1.0],
                "number_of_elements": [2],
                "feat_a": [3.0],
            }
        )
        df.to_csv(csv, index=False)
        loader = DataLoader(train_path=csv)
        with pytest.raises(ValueError, match="columns"):
            loader.load()

    def test_raises_before_load(self, tmp_path) -> None:
        loader = DataLoader(train_path=tmp_path / "train.csv")
        with pytest.raises(RuntimeError, match="Call .load()"):
            _ = loader.train

    def test_manifest_contains_sha256(self, tmp_path) -> None:
        csv = tmp_path / "train.csv"
        _write_valid_csv(csv)
        loader = DataLoader(train_path=csv).load()
        manifest = loader.build_manifest()
        assert len(manifest.sha256) == 64, "SHA-256 should be 64 hex characters."

    def test_manifest_row_count(self, tmp_path) -> None:
        csv = tmp_path / "train.csv"
        _write_valid_csv(csv)
        loader = DataLoader(train_path=csv).load()
        manifest = loader.build_manifest()
        assert manifest.row_count == 10

    def test_manifest_roundtrip(self, tmp_path) -> None:
        csv = tmp_path / "train.csv"
        _write_valid_csv(csv)
        loader = DataLoader(train_path=csv).load()
        manifest = loader.build_manifest()
        out_path = tmp_path / "manifest.json"
        manifest.save(out_path)
        loaded = DataManifest.load(out_path)
        assert loaded.sha256 == manifest.sha256
        assert loaded.row_count == manifest.row_count

    def test_metadata_is_none_when_not_provided(self, tmp_path) -> None:
        csv = tmp_path / "train.csv"
        _write_valid_csv(csv)
        loader = DataLoader(train_path=csv).load()
        assert loader.metadata is None

    def test_metadata_loaded_when_provided(self, tmp_path) -> None:
        train_csv = tmp_path / "train.csv"
        meta_csv = tmp_path / "meta.csv"
        _write_valid_csv(train_csv)
        pd.DataFrame({"formula": ["YBCO"], "material": ["cuprate"]}).to_csv(meta_csv, index=False)
        loader = DataLoader(train_path=train_csv, metadata_path=meta_csv).load()
        assert loader.metadata is not None
        assert "formula" in loader.metadata.columns
