"""
src/monitoring/drift.py
────────────────────────
Data drift detection for the superconductivity feature set.

Strategy: Kolmogorov-Smirnov two-sample test per feature.
• Statistically principled for continuous distributions (no assumption of normality).
• O(n log n) — fast enough to run on every inference batch.

Usage
─────
    monitor = DriftMonitor(reference_data=train_X, significance_level=0.05)
    report  = monitor.detect(new_batch_X)
    if report.drift_detected:
        alert(report.drifted_features)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger
from scipy import stats


# ─── Report ───────────────────────────────────────────────────────────────────


@dataclass
class DriftReport:
    """Summary of drift detection results across all monitored features."""

    significance_level: float
    feature_results: dict[str, dict[str, float]] = field(default_factory=dict)
    """
    Keyed by feature name. Each value is a dict with keys:
      • ks_statistic  — KS test statistic (0 = identical, 1 = maximally different)
      • p_value       — p-value of the test
      • drifted       — True if p_value < significance_level
    """

    @property
    def drift_detected(self) -> bool:
        return any(v["drifted"] for v in self.feature_results.values())

    @property
    def drifted_features(self) -> list[str]:
        return [f for f, v in self.feature_results.items() if v["drifted"]]

    def to_dict(self) -> dict:
        return {
            "significance_level": self.significance_level,
            "overall_drift_detected": self.drift_detected,
            "n_features_drifted": len(self.drifted_features),
            "feature_results": self.feature_results,
        }

    def summary(self) -> str:
        total = len(self.feature_results)
        n_drifted = len(self.drifted_features)
        status = "⚠️  DRIFT DETECTED" if self.drift_detected else "✅  No drift"
        return (
            f"{status} — {n_drifted}/{total} features exceed "
            f"α={self.significance_level}\n"
            + (f"Drifted: {self.drifted_features}" if n_drifted else "")
        )


# ─── Monitor ──────────────────────────────────────────────────────────────────


class DriftMonitor:
    """
    Monitors for distribution shift between a reference dataset and new data.

    Parameters
    ----------
    reference_data : pd.DataFrame
        Held-out training data to compare against.
    significance_level : float
        KS-test p-value threshold below which drift is flagged (default 0.05).
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        significance_level: float = 0.05,
    ) -> None:
        self.reference_data = reference_data.copy()
        self.significance_level = significance_level
        self._numeric_columns = reference_data.select_dtypes(include="number").columns.tolist()

    def detect(self, new_data: pd.DataFrame) -> DriftReport:
        """
        Run KS tests for all shared numeric features.

        Parameters
        ----------
        new_data : pd.DataFrame
            Incoming batch to test (must share columns with reference_data).

        Returns
        -------
        DriftReport
        """
        report = DriftReport(significance_level=self.significance_level)
        shared_cols = [
            c for c in self._numeric_columns if c in new_data.columns
        ]

        if not shared_cols:
            logger.warning("No shared numeric columns found for drift detection.")
            return report

        for col in shared_cols:
            ref_vals = self.reference_data[col].dropna().values
            new_vals = new_data[col].dropna().values

            if len(new_vals) < 5:
                logger.debug(f"Skipping '{col}' — insufficient new samples ({len(new_vals)}).")
                continue

            ks_stat, p_value = stats.ks_2samp(ref_vals, new_vals)
            drifted = bool(p_value < self.significance_level)

            report.feature_results[col] = {
                "ks_statistic": round(float(ks_stat), 6),
                "p_value": round(float(p_value), 6),
                "drifted": drifted,
            }

        logger.info(report.summary())
        return report
