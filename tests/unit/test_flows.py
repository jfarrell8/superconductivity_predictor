"""
tests/unit/test_flows.py
─────────────────────────
Unit tests for individual Prefect tasks.

Prefect 3 tasks can be called directly as regular functions in tests
(without a running server) — just call task.fn(...) or the task itself.
This tests the business logic inside each task without orchestration overhead.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_cfg(tmp_path: Path) -> dict:
    """Build a minimal config dict pointing at tmp_path."""
    raw_train = tmp_path / "data/raw/train.csv"
    processed = tmp_path / "data/processed/train_reduced.csv"
    raw_train.parent.mkdir(parents=True, exist_ok=True)
    processed.parent.mkdir(parents=True, exist_ok=True)

    # Write a minimal valid superconductivity-like CSV
    rng = np.random.default_rng(0)
    n, n_feat = 200, 79
    feat_cols = {f"feat_{i}": rng.uniform(0, 100, size=n) for i in range(n_feat)}
    df = pd.DataFrame(
        {
            "critical_temp": rng.uniform(1.0, 140.0, size=n),
            "number_of_elements": rng.integers(1, 10, size=n),
            "range_Valence": rng.integers(0, 7, size=n),
            **feat_cols,
        }
    )
    df.to_csv(raw_train, index=False)

    return {
        "experiment": {"name": "test", "random_seed": 42},
        "data": {
            "raw_train_path": str(raw_train),
            "processed_path": str(processed),
        },
        "feature_engineering": {
            "target_column": "critical_temp",
            "engineered_target_column": "critical_temp_boxcox",
            "element_bins": {
                "column": "number_of_elements",
                "low_threshold": 1, "low_value": 2,
                "high_threshold": 7, "high_value": 6,
            },
            "valence_bins": {
                "column": "range_Valence",
                "merge_zero_to": 1, "high_threshold": 4, "high_value": 4,
            },
            "protected_columns": ["num_elements_simplified"],
            "correlation_threshold": 0.8,
            "drop_low_mi_columns": [
                "rangeValence_1", "rangeValence_2", "rangeValence_3", "rangeValence_4"
            ],
        },
        "preprocessing": {"columns_to_skip_scaling": ["num_elements_simplified"], "test_size": 0.2},
        "modeling": {
            "n_trials": 2,
            "cv_folds": 2,
            "n_jobs": 1,
            "search_space": ["lightgbm"],
        },
        "evaluation": {
            "feature_counts": [5, 3],
            "final_top_k_features": 3,
        },
        "artifacts": {
            "model_dir": str(tmp_path / "models"),
            "model_filename": "best_model.pkl",
            "top_k_model_filename": "best_model_top3.pkl",
        },
        "storage": {"backend": "local"},
    }


# ─── ingest_data task ─────────────────────────────────────────────────────────


class TestIngestDataTask:
    def test_returns_dataframe_and_manifest(self, tmp_path: Path) -> None:
        from src.orchestration.flows import ingest_data
        cfg = _make_cfg(tmp_path)
        # Call the underlying function directly (bypasses Prefect task wrapper)
        df, manifest = ingest_data.fn(cfg)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert manifest.row_count == len(df)

    def test_manifest_has_valid_sha256(self, tmp_path: Path) -> None:
        from src.orchestration.flows import ingest_data
        cfg = _make_cfg(tmp_path)
        _, manifest = ingest_data.fn(cfg)
        assert len(manifest.sha256) == 64


# ─── engineer_features task ───────────────────────────────────────────────────


class TestEngineerFeaturesTask:
    def test_returns_x_and_y(self, tmp_path: Path) -> None:
        from src.orchestration.flows import engineer_features, ingest_data
        cfg = _make_cfg(tmp_path)
        raw_df, _ = ingest_data.fn(cfg)
        X, y = engineer_features.fn(raw_df, cfg)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)

    def test_target_not_in_features(self, tmp_path: Path) -> None:
        from src.orchestration.flows import engineer_features, ingest_data
        cfg = _make_cfg(tmp_path)
        raw_df, _ = ingest_data.fn(cfg)
        X, _ = engineer_features.fn(raw_df, cfg)
        assert "critical_temp_boxcox" not in X.columns
        assert "critical_temp" not in X.columns

    def test_processed_csv_written(self, tmp_path: Path) -> None:
        from src.orchestration.flows import engineer_features, ingest_data
        cfg = _make_cfg(tmp_path)
        raw_df, _ = ingest_data.fn(cfg)
        engineer_features.fn(raw_df, cfg)
        assert Path(cfg["data"]["processed_path"]).exists()


# ─── split_data task ──────────────────────────────────────────────────────────


class TestSplitDataTask:
    def test_split_sizes(self, tmp_path: Path) -> None:
        from src.orchestration.flows import engineer_features, ingest_data, split_data
        cfg = _make_cfg(tmp_path)
        raw_df, _ = ingest_data.fn(cfg)
        X, y = engineer_features.fn(raw_df, cfg)
        X_train, X_test, y_train, y_test = split_data.fn(X, y, cfg)
        total = len(X_train) + len(X_test)
        assert total == len(X)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_no_index_overlap(self, tmp_path: Path) -> None:
        from src.orchestration.flows import engineer_features, ingest_data, split_data
        cfg = _make_cfg(tmp_path)
        raw_df, _ = ingest_data.fn(cfg)
        X, y = engineer_features.fn(raw_df, cfg)
        X_train, X_test, _, _ = split_data.fn(X, y, cfg)
        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0


# ─── Full flow smoke test (no Prefect server) ─────────────────────────────────


class TestTrainingPipelineSmoke:
    def test_pipeline_returns_expected_keys(self, tmp_path: Path) -> None:
        """
        Run the full flow in-process with 2 trials.
        Verifies the contract: result dict has the expected keys and types.
        """
        from src.orchestration.flows import training_pipeline

        cfg = _make_cfg(tmp_path)
        # Write config to a temp YAML file so the flow can load it
        import yaml
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(cfg))

        result = training_pipeline(
            config_path=str(config_path),
            n_trials_override=2,
        )

        assert "model_type" in result
        assert "cv_rmse" in result
        assert "test_rmse" in result
        assert "test_r2" in result
        assert "top_features" in result
        assert isinstance(result["top_features"], list)
        assert isinstance(result["cv_rmse"], float)
        assert result["cv_rmse"] > 0
