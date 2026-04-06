"""
src/evaluation/evaluator.py
────────────────────────────
Model evaluation: standard regression metrics, SHAP-based explainability,
and the feature-reduction sweep from the modeling notebook.

All methods return plain Python/pandas objects so they're easy to log,
serialize, or render in a dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── Metric container ─────────────────────────────────────────────────────────


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float
    n_samples: int

    def __str__(self) -> str:
        return f"RMSE={self.rmse:.4f}  MAE={self.mae:.4f}  R²={self.r2:.4f}  n={self.n_samples}"

    def to_dict(self) -> dict[str, float | int]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "n_samples": self.n_samples,
        }


# ─── Evaluator ────────────────────────────────────────────────────────────────


class ModelEvaluator:
    """
    Computes metrics, feature importances, SHAP values, and the feature-count
    reduction sweep described in the modeling notebook.

    Parameters
    ----------
    model : BaseEstimator
        A fitted sklearn-compatible regressor.
    """

    def __init__(self, model: BaseEstimator) -> None:
        self.model = model
        self._shap_values: np.ndarray | None = None
        self._shap_explainer: shap.TreeExplainer | None = None

    # ── Core metrics ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        label: str = "Evaluation",
    ) -> RegressionMetrics:
        """Compute and log RMSE, MAE, and R² on a held-out split."""
        y_pred = self.model.predict(X)
        metrics = RegressionMetrics(
            rmse=float(np.sqrt(mean_squared_error(y, y_pred))),
            mae=float(mean_absolute_error(y, y_pred)),
            r2=float(r2_score(y, y_pred)),
            n_samples=len(y),
        )
        logger.info(f"{label}: {metrics}")
        return metrics

    def log_artifacts_to_mlflow(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        top_features: list[str],
        run_id: str | None = None,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mlflow

        # Check MLflow server is reachable
        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri and tracking_uri.startswith("http"):
            import urllib.request

            try:
                urllib.request.urlopen(tracking_uri, timeout=3)
            except Exception:
                logger.warning("MLflow server not reachable — skipping artifact logging.")
                return

        try:
            ctx = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run()
        except Exception as exc:
            logger.warning(f"Could not start MLflow run: {exc}")
            return

        with ctx:
            y_pred = self.model.predict(X_test[top_features])

            # ── 1. Actual vs predicted ────────────────────────────────────
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_test, y_pred, alpha=0.3, s=8, color="#185FA5")
            lims = [
                min(y_test.min(), y_pred.min()),
                max(y_test.max(), y_pred.max()),
            ]
            ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
            ax.set_xlabel("Actual Tc (Box-Cox scale)")
            ax.set_ylabel("Predicted Tc (Box-Cox scale)")
            ax.set_title("Actual vs predicted critical temperature")
            ax.legend(fontsize=9)
            fig.tight_layout()
            mlflow.log_figure(fig, "plots/actual_vs_predicted.png")
            plt.close(fig)

            # ── 2. Residuals plot ─────────────────────────────────────────
            residuals = y_test.values - y_pred
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(y_pred, residuals, alpha=0.3, s=8, color="#534AB7")
            ax.axhline(0, color="red", linestyle="--", linewidth=1)
            ax.set_xlabel("Predicted Tc (Box-Cox scale)")
            ax.set_ylabel("Residual (actual − predicted)")
            ax.set_title("Residuals vs predicted")
            ax.grid(True, linestyle="--", alpha=0.3)
            fig.tight_layout()
            mlflow.log_figure(fig, "plots/residuals.png")
            plt.close(fig)

            # ── 3. Residual distribution ──────────────────────────────────
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(residuals, bins=50, color="#854F0B", alpha=0.7, edgecolor="none")
            ax.axvline(0, color="red", linestyle="--", linewidth=1)
            ax.set_xlabel("Residual")
            ax.set_ylabel("Count")
            ax.set_title(
                f"Residual distribution  (mean={residuals.mean():.3f}, std={residuals.std():.3f})"
            )
            fig.tight_layout()
            mlflow.log_figure(fig, "plots/residual_distribution.png")
            plt.close(fig)

            # ── 4. Feature importance bar chart ───────────────────────────
            try:
                imp_df = self.feature_importances(top_features)
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.barh(
                    imp_df["feature"][::-1],
                    imp_df["importance"][::-1],
                    color="#1D9E75",
                    height=0.6,
                )
                ax.set_xlabel("Feature importance")
                ax.set_title(f"Top {len(top_features)} feature importances")
                ax.grid(True, axis="x", linestyle="--", alpha=0.4)
                fig.tight_layout()
                mlflow.log_figure(fig, "plots/feature_importance.png")
                plt.close(fig)
            except Exception as exc:
                logger.warning(f"Feature importance plot failed (non-fatal): {exc}")

            # ── 5. SHAP summary plot ──────────────────────────────────────
            try:
                shap_vals = self.compute_shap(X_test[top_features])
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.summary_plot(
                    shap_vals,
                    X_test[top_features],
                    show=False,
                    plot_size=None,
                )
                fig = plt.gcf()
                mlflow.log_figure(fig, "plots/shap_summary.png")
                plt.close(fig)
                logger.info("MLflow: logged SHAP summary plot")
            except Exception as exc:
                logger.warning(f"SHAP plot failed (non-fatal): {exc}")

            logger.info("MLflow: all evaluation artifacts logged")

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importances(self, feature_names: list[str]) -> pd.DataFrame:
        """
        Extract feature importances (tree models) or coefficient magnitudes
        (linear models), sorted descending.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_)
        else:
            raise AttributeError(
                f"Model of type {type(self.model).__name__} has no "
                "feature_importances_ or coef_ attribute."
            )

        return (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ── SHAP ─────────────────────────────────────────────────────────────────

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values using TreeExplainer (supports XGB, LGBM, RF).

        Falls back to KernelExplainer for linear / unsupported models.
        """
        try:
            self._shap_explainer = shap.TreeExplainer(self.model)
            self._shap_values = self._shap_explainer.shap_values(X)
        except Exception:
            logger.warning(
                "TreeExplainer not compatible with this model type; "
                "falling back to KernelExplainer (slower)."
            )
            self._shap_explainer = shap.KernelExplainer(  # type: ignore[assignment]
                self.model.predict, shap.sample(X, 100)
            )
            self._shap_values = self._shap_explainer.shap_values(X)

        return self._shap_values

    def plot_shap_summary(self, X: pd.DataFrame) -> plt.Figure:
        """Generate and return a SHAP summary plot figure."""
        if self._shap_values is None:
            self.compute_shap(X)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(self._shap_values, X, show=False)
        fig = plt.gcf()
        return fig

    # ── Feature reduction sweep ────────────────────────────────────────────────

    def feature_reduction_sweep(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_counts: list[int],
        importance_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Train the same model architecture with decreasing feature subsets
        and return a DataFrame of (n_features, rmse) for elbow-curve analysis.

        Mirrors the notebook's feature_counts sweep, enabling the lab-scientist
        use-case: fewer measurements → faster iteration with minimal accuracy loss.
        """
        if importance_df is None:
            importance_df = self.feature_importances(X_train.columns.tolist())

        results = []
        for k in feature_counts:
            top_k = importance_df.head(k)["feature"].tolist()
            self.model.fit(X_train[top_k], y_train)
            preds = self.model.predict(X_test[top_k])
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            results.append({"n_features": k, "rmse": rmse})
            logger.debug(f"  k={k:>3}  RMSE={rmse:.4f}")

        df = pd.DataFrame(results)
        logger.info(f"Feature reduction sweep:\n{df.to_string(index=False)}")
        return df

    def plot_reduction_curve(self, reduction_df: pd.DataFrame) -> plt.Figure:
        """Plot the elbow curve from a feature reduction sweep."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            reduction_df["n_features"],
            reduction_df["rmse"],
            marker="o",
            linewidth=2,
            color="steelblue",
        )
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("Test RMSE (Box-Cox scale)")
        ax.set_title("Feature Reduction vs. Model Performance")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        return fig

    def top_k_features(self, importance_df: pd.DataFrame, k: int) -> list[str]:
        """Return the names of the top k features by importance."""
        return importance_df.head(k)["feature"].tolist()


class _null_context:
    def __enter__(self) -> _null_context:
        return self

    def __exit__(self, *_: Any) -> None:
        pass
