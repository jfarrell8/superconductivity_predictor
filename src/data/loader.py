"""
src/data/loader.py
──────────────────
Responsible for ingesting raw CSV data, validating schema and integrity,
and surfacing a clean DataFrame for downstream processing.

Design goals
────────────
• Single responsibility: loading + validation only, no transforms here.
• Fail fast: raise informative errors rather than silently propagating bad data.
• Type-safe: all public methods are fully annotated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

# ─── Schema definition ───────────────────────────────────────────────────────

# Minimum required columns that must be present in the raw training file.
# The full 81-feature set is validated by count rather than name to stay
# flexible if the source dataset adds/removes minor columns in future versions.
REQUIRED_COLUMNS: frozenset[str] = frozenset(["critical_temp", "number_of_elements"])
MIN_EXPECTED_FEATURE_COUNT: int = 80  # target + 81 features - overhead


@dataclass
class DataManifest:
    """Lightweight provenance record written alongside processed files."""

    source_path: str
    row_count: int
    column_count: int
    sha256: str
    missing_values: dict[str, int] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2))
        logger.info(f"Manifest written to {path}")

    @classmethod
    def load(cls, path: Path) -> DataManifest:
        data = json.loads(path.read_text())
        return cls(**data)


# ─── Loader ──────────────────────────────────────────────────────────────────


class DataLoader:
    """
    Loads and validates the superconductivity dataset from disk.

    Parameters
    ----------
    train_path : Path | str
        Path to the raw training CSV (e.g. data/raw/train.csv).
    metadata_path : Path | str | None
        Optional path to the compound metadata CSV (unique_m.csv).
    """

    def __init__(
        self,
        train_path: Path | str,
        metadata_path: Path | str | None = None,
    ) -> None:
        self.train_path = Path(train_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self._train_df: pd.DataFrame | None = None
        self._metadata_df: pd.DataFrame | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self) -> DataLoader:
        """Load both CSVs from disk, validate, and cache internally."""
        logger.info(f"Loading training data from {self.train_path}")
        self._train_df = self._read_csv(self.train_path)
        self._validate_train(self._train_df)

        if self.metadata_path:
            logger.info(f"Loading metadata from {self.metadata_path}")
            self._metadata_df = self._read_csv(self.metadata_path)

        return self  # fluent interface

    @property
    def train(self) -> pd.DataFrame:
        """Return the validated training DataFrame."""
        if self._train_df is None:
            raise RuntimeError("Call .load() before accessing .train")
        return self._train_df.copy()

    @property
    def metadata(self) -> pd.DataFrame | None:
        """Return the metadata DataFrame if loaded."""
        return self._metadata_df.copy() if self._metadata_df is not None else None

    def build_manifest(self) -> DataManifest:
        """Generate a provenance manifest for the loaded training data."""
        df = self.train
        sha = self._file_sha256(self.train_path)
        missing = df.isna().sum()
        return DataManifest(
            source_path=str(self.train_path.resolve()),
            row_count=len(df),
            column_count=len(df.columns),
            sha256=sha,
            missing_values={k: int(v) for k, v in missing.items() if v > 0},
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        df = pd.read_csv(path)
        logger.debug(f"Loaded {len(df):,} rows × {len(df.columns)} columns from {path.name}")
        return df

    @staticmethod
    def _validate_train(df: pd.DataFrame) -> None:
        """Raise ValueError if the DataFrame fails any integrity check."""
        missing_cols = REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Training data missing required columns: {missing_cols}")

        if len(df.columns) < MIN_EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected ≥{MIN_EXPECTED_FEATURE_COUNT} columns, "
                f"got {len(df.columns)}. Possible truncated file."
            )

        if df.empty:
            raise ValueError("Training DataFrame is empty.")

        null_counts = df.isna().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            logger.warning(
                f"Null values detected in columns: {cols_with_nulls.to_dict()}. "
                "Consider imputation before feature engineering."
            )

        non_numeric = df.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            logger.warning(f"Non-numeric columns detected: {non_numeric}")

        logger.info("Training data validation passed.")

    @staticmethod
    def _file_sha256(path: Path) -> str:
        """Compute SHA-256 hash of a file for data versioning."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
