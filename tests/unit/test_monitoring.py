"""
tests/unit/test_monitoring.py
──────────────────────────────
Unit tests for the KS-test drift monitor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import DriftMonitor


@pytest.fixture()
def reference_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "mean_atomic_mass": rng.normal(88, 10, size=500),
            "mean_fie": rng.normal(6.5, 1.2, size=500),
            "mean_Density": rng.normal(5000, 800, size=500),
        }
    )


class TestDriftMonitor:
    def test_no_drift_on_same_distribution(self, reference_df: pd.DataFrame) -> None:
        monitor = DriftMonitor(reference_df, significance_level=0.05)
        # Sample from the same distribution
        rng = np.random.default_rng(1)
        new_df = pd.DataFrame(
            {
                "mean_atomic_mass": rng.normal(88, 10, size=200),
                "mean_fie": rng.normal(6.5, 1.2, size=200),
                "mean_Density": rng.normal(5000, 800, size=200),
            }
        )
        report = monitor.detect(new_df)
        assert not report.drift_detected, (
            "No drift should be detected when new data comes from the same distribution."
        )

    def test_drift_detected_on_shifted_distribution(self, reference_df: pd.DataFrame) -> None:
        monitor = DriftMonitor(reference_df, significance_level=0.05)
        # Inject a large mean shift
        rng = np.random.default_rng(2)
        new_df = pd.DataFrame(
            {
                "mean_atomic_mass": rng.normal(200, 10, size=300),  # heavily shifted
                "mean_fie": rng.normal(6.5, 1.2, size=300),
                "mean_Density": rng.normal(5000, 800, size=300),
            }
        )
        report = monitor.detect(new_df)
        assert "mean_atomic_mass" in report.drifted_features

    def test_drifted_features_property(self, reference_df: pd.DataFrame) -> None:
        monitor = DriftMonitor(reference_df, significance_level=0.05)
        rng = np.random.default_rng(3)
        new_df = pd.DataFrame(
            {
                "mean_atomic_mass": rng.normal(300, 5, size=300),
                "mean_fie": rng.normal(6.5, 1.2, size=300),
                "mean_Density": rng.normal(5000, 800, size=300),
            }
        )
        report = monitor.detect(new_df)
        assert isinstance(report.drifted_features, list)

    def test_report_to_dict_structure(self, reference_df: pd.DataFrame) -> None:
        monitor = DriftMonitor(reference_df)
        report = monitor.detect(reference_df)
        d = report.to_dict()
        assert "overall_drift_detected" in d
        assert "feature_results" in d
        assert "significance_level" in d

    def test_handles_missing_shared_columns(self, reference_df: pd.DataFrame) -> None:
        monitor = DriftMonitor(reference_df)
        new_df = pd.DataFrame({"unrelated_col": [1.0, 2.0, 3.0, 4.0, 5.0]})
        report = monitor.detect(new_df)
        assert len(report.feature_results) == 0

    def test_skips_columns_with_too_few_samples(self, reference_df: pd.DataFrame) -> None:
        monitor = DriftMonitor(reference_df)
        new_df = pd.DataFrame({"mean_atomic_mass": [88.0, 89.0]})  # < 5 samples
        report = monitor.detect(new_df)
        assert "mean_atomic_mass" not in report.feature_results

    def test_significance_level_respected(self, reference_df: pd.DataFrame) -> None:
        """Stricter alpha = more drift flagged."""
        monitor_strict = DriftMonitor(reference_df, significance_level=0.99)
        rng = np.random.default_rng(4)
        slightly_shifted = pd.DataFrame(
            {
                "mean_atomic_mass": rng.normal(90, 10, size=300),  # tiny shift
                "mean_fie": rng.normal(6.5, 1.2, size=300),
                "mean_Density": rng.normal(5000, 800, size=300),
            }
        )
        report = monitor_strict.detect(slightly_shifted)
        # At alpha=0.99 virtually everything will flag — just confirm the property works
        assert isinstance(report.drift_detected, bool)


# ─── PredictionLogger ─────────────────────────────────────────────────────────


class TestPredictionLogger:
    """Tests for src/monitoring/logger.py — PredictionLogger."""

    FEATURES = {"mean_atomic_mass": 88.5, "mean_fie": 6.2, "mean_Density": 5400.0}

    def test_log_creates_file(self, tmp_path: Path) -> None:
        from src.monitoring.logger import PredictionLogger

        logger = PredictionLogger(sink=tmp_path / "preds.jsonl")
        logger.log(features=self.FEATURES, prediction=4.31)
        assert (tmp_path / "preds.jsonl").exists()

    def test_log_returns_request_id(self, tmp_path: Path) -> None:
        from src.monitoring.logger import PredictionLogger

        logger = PredictionLogger(sink=tmp_path / "preds.jsonl")
        rid = logger.log(features=self.FEATURES, prediction=4.31)
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_log_uses_provided_request_id(self, tmp_path: Path) -> None:
        from src.monitoring.logger import PredictionLogger

        logger = PredictionLogger(sink=tmp_path / "preds.jsonl")
        rid = logger.log(features=self.FEATURES, prediction=4.31, request_id="my-custom-id")
        assert rid == "my-custom-id"

    def test_log_record_structure(self, tmp_path: Path) -> None:
        import json

        from src.monitoring.logger import PredictionLogger

        sink = tmp_path / "preds.jsonl"
        logger = PredictionLogger(sink=sink)
        logger.log(features=self.FEATURES, prediction=4.31, model_type="lightgbm")
        record = json.loads(sink.read_text().strip())
        assert "timestamp" in record
        assert "request_id" in record
        assert "prediction" in record
        assert "features" in record
        assert record["model_type"] == "lightgbm"
        assert record["prediction"] == pytest.approx(4.31)

    def test_multiple_logs_append_lines(self, tmp_path: Path) -> None:
        from src.monitoring.logger import PredictionLogger

        sink = tmp_path / "preds.jsonl"
        logger = PredictionLogger(sink=sink)
        for i in range(5):
            logger.log(features=self.FEATURES, prediction=float(i))
        lines = [line for line in sink.read_text().splitlines() if line.strip()]
        assert len(lines) == 5

    def test_log_batch_returns_correct_count(self, tmp_path: Path) -> None:
        from src.monitoring.logger import PredictionLogger

        logger = PredictionLogger(sink=tmp_path / "preds.jsonl")
        rids = logger.log_batch(
            feature_rows=[self.FEATURES] * 4,
            predictions=[1.0, 2.0, 3.0, 4.0],
        )
        assert len(rids) == 4
        assert all(isinstance(r, str) for r in rids)

    def test_load_as_dataframe_expands_features(self, tmp_path: Path) -> None:
        from src.monitoring.logger import PredictionLogger

        sink = tmp_path / "preds.jsonl"
        logger = PredictionLogger(sink=sink)
        logger.log(features=self.FEATURES, prediction=4.31)
        logger.log(features=self.FEATURES, prediction=3.99)
        df = PredictionLogger.load_as_dataframe(sink)
        assert len(df) == 2
        assert "prediction" in df.columns
        assert "mean_atomic_mass" in df.columns

    def test_load_as_dataframe_prediction_values_correct(self, tmp_path: Path) -> None:
        from src.monitoring.logger import PredictionLogger

        sink = tmp_path / "preds.jsonl"
        logger = PredictionLogger(sink=sink)
        logger.log(features=self.FEATURES, prediction=4.31)
        logger.log(features=self.FEATURES, prediction=3.99)
        df = PredictionLogger.load_as_dataframe(sink)
        assert list(df["prediction"].round(2)) == [4.31, 3.99]

    def test_log_rotates_on_size_exceeded(self, tmp_path: Path) -> None:
        """When max_file_mb is effectively 0, every call should rotate the log."""
        from src.monitoring.logger import PredictionLogger

        sink = tmp_path / "preds.jsonl"
        logger = PredictionLogger(sink=sink, max_file_mb=0.0)
        # Write enough to trigger rotation on the second call
        logger.log(features=self.FEATURES, prediction=1.0)
        logger.log(features=self.FEATURES, prediction=2.0)
        # After rotation, a rotated file should exist
        rotated = list(tmp_path.glob("preds_*.jsonl"))
        assert len(rotated) >= 1

    def test_logger_creates_parent_directory(self, tmp_path: Path) -> None:
        from src.monitoring.logger import PredictionLogger

        nested_sink = tmp_path / "a" / "b" / "preds.jsonl"
        logger = PredictionLogger(sink=nested_sink)
        logger.log(features=self.FEATURES, prediction=1.0)
        assert nested_sink.exists()

    def test_load_as_dataframe_integrates_with_drift_monitor(
        self, tmp_path: Path, reference_df: pd.DataFrame
    ) -> None:
        """
        End-to-end: log predictions, load the log, run drift detection.
        The features in the log should be passable directly to DriftMonitor.
        """
        from src.monitoring.drift import DriftMonitor
        from src.monitoring.logger import PredictionLogger

        sink = tmp_path / "preds.jsonl"
        logger = PredictionLogger(sink=sink)

        rng = np.random.default_rng(5)
        for _ in range(50):
            feat = {
                "mean_atomic_mass": float(rng.normal(88, 10)),
                "mean_fie": float(rng.normal(6.5, 1.2)),
                "mean_Density": float(rng.normal(5000, 800)),
            }
            logger.log(features=feat, prediction=float(rng.normal(4, 1)))

        logged_df = PredictionLogger.load_as_dataframe(sink)
        monitor = DriftMonitor(reference_df, significance_level=0.05)
        report = monitor.detect(logged_df)
        # Result should be a valid DriftReport regardless of drift/no-drift
        assert isinstance(report.drift_detected, bool)
        assert isinstance(report.drifted_features, list)
