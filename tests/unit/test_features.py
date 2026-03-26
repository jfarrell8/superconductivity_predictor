"""
tests/unit/test_features.py
────────────────────────────
Unit tests for every transformer in src/features/engineer.py.
Each test is isolated — no filesystem access, no model training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.features.engineer import (
    BinningConfig,
    BinningTransformer,
    FeaturePruner,
    FeatureScaler,
    PrunerConfig,
    TargetTransformer,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def small_df() -> pd.DataFrame:
    """Minimal DataFrame that mirrors the superconductivity schema."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "number_of_elements": rng.integers(1, 10, size=n),
            "range_Valence": rng.integers(0, 7, size=n),
            "mean_atomic_mass": rng.uniform(1, 200, size=n),
            "wtd_mean_atomic_mass": rng.uniform(1, 200, size=n),
            "mean_fie": rng.uniform(3, 15, size=n),
            "critical_temp": rng.uniform(1, 140, size=n),  # strictly positive
        }
    )


@pytest.fixture()
def element_binner() -> BinningTransformer:
    return BinningTransformer(
        configs=[
            BinningConfig(
                column="number_of_elements",
                output_column="num_elements_simplified",
                rules=[
                    ("lte", 1, 2),
                    ("gte", 7, 6),
                ],
            )
        ]
    )


# ─── BinningTransformer ───────────────────────────────────────────────────────


class TestBinningTransformer:
    def test_lte_rule_merges_low_values(
        self, element_binner: BinningTransformer, small_df: pd.DataFrame
    ) -> None:
        out = element_binner.fit_transform(small_df)
        # After binning: values <= 1 should become 2
        raw_lows = small_df["number_of_elements"] <= 1
        assert (out.loc[raw_lows, "num_elements_simplified"] == 2).all()

    def test_gte_rule_merges_high_values(
        self, element_binner: BinningTransformer, small_df: pd.DataFrame
    ) -> None:
        out = element_binner.fit_transform(small_df)
        raw_highs = small_df["number_of_elements"] >= 7
        assert (out.loc[raw_highs, "num_elements_simplified"] == 6).all()

    def test_middle_values_unchanged(
        self, element_binner: BinningTransformer, small_df: pd.DataFrame
    ) -> None:
        out = element_binner.fit_transform(small_df)
        middle_mask = (small_df["number_of_elements"] > 1) & (
            small_df["number_of_elements"] < 7
        )
        expected = small_df.loc[middle_mask, "number_of_elements"]
        actual = out.loc[middle_mask, "num_elements_simplified"]
        pd.testing.assert_series_equal(
            expected.reset_index(drop=True),
            actual.reset_index(drop=True),
            check_names=False,
        )

    def test_original_column_preserved(
        self, element_binner: BinningTransformer, small_df: pd.DataFrame
    ) -> None:
        out = element_binner.fit_transform(small_df)
        assert "number_of_elements" in out.columns, (
            "BinningTransformer should not drop the source column."
        )

    def test_missing_column_is_skipped_gracefully(
        self, element_binner: BinningTransformer
    ) -> None:
        df = pd.DataFrame({"some_other_col": [1, 2, 3]})
        out = element_binner.fit_transform(df)  # should not raise
        assert "num_elements_simplified" not in out.columns

    def test_invalid_operator_raises(self) -> None:
        binner = BinningTransformer(
            configs=[
                BinningConfig(
                    column="x",
                    output_column="x_out",
                    rules=[("invalid_op", 5, 6)],
                )
            ]
        )
        df = pd.DataFrame({"x": [1, 5, 10]})
        with pytest.raises(ValueError, match="Unknown operator"):
            binner.transform(df)


# ─── TargetTransformer ────────────────────────────────────────────────────────


class TestTargetTransformer:
    def test_fit_stores_lambda(self, small_df: pd.DataFrame) -> None:
        tt = TargetTransformer("critical_temp", "critical_temp_boxcox")
        tt.fit(small_df)
        assert isinstance(tt.fitted_lambda, float)

    def test_transform_creates_output_column(self, small_df: pd.DataFrame) -> None:
        tt = TargetTransformer("critical_temp", "critical_temp_boxcox")
        out = tt.fit_transform(small_df)
        assert "critical_temp_boxcox" in out.columns

    def test_transform_removes_source_column(self, small_df: pd.DataFrame) -> None:
        tt = TargetTransformer("critical_temp", "critical_temp_boxcox")
        out = tt.fit_transform(small_df)
        assert "critical_temp" not in out.columns

    def test_inverse_transform_roundtrip(self, small_df: pd.DataFrame) -> None:
        tt = TargetTransformer("critical_temp", "critical_temp_boxcox")
        out = tt.fit_transform(small_df)
        reconstructed = tt.inverse_transform(out["critical_temp_boxcox"].values)
        original = small_df["critical_temp"].values
        np.testing.assert_allclose(reconstructed, original, rtol=1e-5)

    def test_raises_if_not_fitted(self) -> None:
        tt = TargetTransformer("critical_temp", "critical_temp_boxcox")
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = tt.fitted_lambda

    def test_raises_on_non_positive_values(self) -> None:
        tt = TargetTransformer("critical_temp", "critical_temp_boxcox")
        df = pd.DataFrame({"critical_temp": [-1.0, 0.0, 5.0]})
        with pytest.raises(ValueError, match="strictly positive"):
            tt.fit(df)

    def test_skewness_reduced_after_boxcox(self, small_df: pd.DataFrame) -> None:
        """Box-Cox should reduce skewness of the right-skewed temperature distribution."""
        # Inject a right-skewed target manually
        skewed_df = small_df.copy()
        skewed_df["critical_temp"] = np.exp(
            np.random.default_rng(0).normal(3, 1, size=len(small_df))
        )
        tt = TargetTransformer("critical_temp", "critical_temp_boxcox")
        out = tt.fit_transform(skewed_df)
        original_skew = abs(stats.skew(skewed_df["critical_temp"]))
        transformed_skew = abs(stats.skew(out["critical_temp_boxcox"]))
        assert transformed_skew < original_skew, (
            "Box-Cox should reduce skewness of right-skewed target."
        )


# ─── FeaturePruner ────────────────────────────────────────────────────────────


class TestFeaturePruner:
    def _make_correlated_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        base = rng.normal(size=300)
        return pd.DataFrame(
            {
                "x1": base,
                "x2": base + rng.normal(scale=0.01, size=300),  # near-perfect correlation
                "x3": rng.normal(size=300),                     # independent
                "target": rng.normal(size=300),
            }
        )

    def test_drops_correlated_column(self) -> None:
        df = self._make_correlated_df()
        pruner = FeaturePruner(PrunerConfig(correlation_threshold=0.8))
        out = pruner.fit_transform(df, target_column="target")
        # x1 and x2 are highly correlated — one of them should be dropped
        assert not ("x1" in out.columns and "x2" in out.columns), (
            "Pruner should remove one of a highly-correlated pair."
        )

    def test_independent_column_retained(self) -> None:
        df = self._make_correlated_df()
        pruner = FeaturePruner(PrunerConfig(correlation_threshold=0.8))
        out = pruner.fit_transform(df, target_column="target")
        assert "x3" in out.columns, "Uncorrelated column should not be dropped."

    def test_explicit_drop_columns(self) -> None:
        df = self._make_correlated_df()
        pruner = FeaturePruner(
            PrunerConfig(correlation_threshold=0.99, drop_columns=["x3"])
        )
        out = pruner.fit_transform(df, target_column="target")
        assert "x3" not in out.columns

    def test_raises_if_not_fitted(self) -> None:
        pruner = FeaturePruner(PrunerConfig())
        with pytest.raises(RuntimeError, match="fit()"):
            pruner.transform(pd.DataFrame({"a": [1, 2]}))


# ─── FeatureScaler ────────────────────────────────────────────────────────────


class TestFeatureScaler:
    def test_scaled_columns_have_zero_mean(self) -> None:
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]}
        )
        scaler = FeatureScaler()
        out = scaler.fit_transform(df)
        np.testing.assert_allclose(out["a"].mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(out["b"].mean(), 0.0, atol=1e-10)

    def test_skip_columns_are_unchanged(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "num_elements_simplified": [2, 3, 4],
            }
        )
        scaler = FeatureScaler(skip_columns=["num_elements_simplified"])
        out = scaler.fit_transform(df)
        pd.testing.assert_series_equal(
            df["num_elements_simplified"], out["num_elements_simplified"]
        )

    def test_transform_without_fit_raises(self) -> None:
        scaler = FeatureScaler()
        with pytest.raises(Exception):
            scaler.transform(pd.DataFrame({"a": [1.0, 2.0]}))
