"""
tests/unit/test_evaluator.py
─────────────────────────────
Unit tests for ModelEvaluator — uses a trivial LinearRegression so tests
run in milliseconds without GPU or heavy model training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.evaluation.evaluator import ModelEvaluator, RegressionMetrics


@pytest.fixture()
def fitted_evaluator() -> tuple[ModelEvaluator, pd.DataFrame, pd.Series]:
    """Returns a ModelEvaluator with a trivial fitted LinearRegression."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(
        {f"feature_{i}": rng.normal(size=n) for i in range(5)}
    )
    # Perfect linear target so we can assert R² ≈ 1
    y = pd.Series(
        X["feature_0"] * 2.5 + X["feature_1"] * -1.3 + rng.normal(scale=0.01, size=n),
        name="target",
    )
    model = LinearRegression()
    model.fit(X, y)
    return ModelEvaluator(model), X, y


class TestRegressionMetrics:
    def test_to_dict_contains_all_keys(self) -> None:
        m = RegressionMetrics(rmse=0.5, mae=0.4, r2=0.9, n_samples=100)
        d = m.to_dict()
        assert set(d.keys()) == {"rmse", "mae", "r2", "n_samples"}

    def test_str_repr(self) -> None:
        m = RegressionMetrics(rmse=0.5, mae=0.4, r2=0.9, n_samples=100)
        assert "RMSE" in str(m) and "R²" in str(m)


class TestModelEvaluator:
    def test_evaluate_returns_metrics(
        self, fitted_evaluator: tuple[ModelEvaluator, pd.DataFrame, pd.Series]
    ) -> None:
        evaluator, X, y = fitted_evaluator
        metrics = evaluator.evaluate(X, y)
        assert isinstance(metrics, RegressionMetrics)

    def test_r2_near_one_for_near_perfect_fit(
        self, fitted_evaluator: tuple[ModelEvaluator, pd.DataFrame, pd.Series]
    ) -> None:
        evaluator, X, y = fitted_evaluator
        metrics = evaluator.evaluate(X, y)
        assert metrics.r2 > 0.99, "R² should be near 1 for near-linear target."

    def test_rmse_non_negative(
        self, fitted_evaluator: tuple[ModelEvaluator, pd.DataFrame, pd.Series]
    ) -> None:
        evaluator, X, y = fitted_evaluator
        metrics = evaluator.evaluate(X, y)
        assert metrics.rmse >= 0

    def test_n_samples_matches_input(
        self, fitted_evaluator: tuple[ModelEvaluator, pd.DataFrame, pd.Series]
    ) -> None:
        evaluator, X, y = fitted_evaluator
        metrics = evaluator.evaluate(X, y)
        assert metrics.n_samples == len(y)

    def test_feature_importances_linear_model(
        self, fitted_evaluator: tuple[ModelEvaluator, pd.DataFrame, pd.Series]
    ) -> None:
        evaluator, X, _ = fitted_evaluator
        imp_df = evaluator.feature_importances(X.columns.tolist())
        assert list(imp_df.columns) == ["feature", "importance"]
        assert len(imp_df) == X.shape[1]
        # Importances should be sorted descending
        assert (imp_df["importance"].diff().dropna() <= 0).all()

    def test_feature_importances_missing_attribute_raises(self) -> None:
        """Models without feature_importances_ or coef_ should raise AttributeError."""
        from unittest.mock import MagicMock
        mock_model = MagicMock(spec=[])  # no attributes at all
        evaluator = ModelEvaluator(mock_model)
        with pytest.raises(AttributeError):
            evaluator.feature_importances(["a", "b"])

    def test_top_k_features_returns_correct_length(
        self, fitted_evaluator: tuple[ModelEvaluator, pd.DataFrame, pd.Series]
    ) -> None:
        evaluator, X, _ = fitted_evaluator
        imp_df = evaluator.feature_importances(X.columns.tolist())
        top3 = evaluator.top_k_features(imp_df, k=3)
        assert len(top3) == 3

    def test_feature_reduction_sweep_shape(
        self, fitted_evaluator: tuple[ModelEvaluator, pd.DataFrame, pd.Series]
    ) -> None:
        evaluator, X, y = fitted_evaluator
        counts = [5, 3, 2]
        imp_df = evaluator.feature_importances(X.columns.tolist())
        result = evaluator.feature_reduction_sweep(
            X_train=X,
            X_test=X,
            y_train=y,
            y_test=y,
            feature_counts=counts,
            importance_df=imp_df,
        )
        assert list(result.columns) == ["n_features", "rmse"]
        assert len(result) == len(counts)

    def test_plot_reduction_curve_returns_figure(
        self, fitted_evaluator: tuple[ModelEvaluator, pd.DataFrame, pd.Series]
    ) -> None:
        import matplotlib.pyplot as plt
        evaluator, X, y = fitted_evaluator
        imp_df = evaluator.feature_importances(X.columns.tolist())
        reduction_df = evaluator.feature_reduction_sweep(
            X, X, y, y, [5, 3], imp_df
        )
        fig = evaluator.plot_reduction_curve(reduction_df)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
