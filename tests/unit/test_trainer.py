"""
tests/unit/test_trainer.py
───────────────────────────
Unit tests for ModelTrainer, ModelRegistry, and MLflow integration.

MLflow tracking is disabled in all tests (mlflow_cfg=None) so tests
run without a tracking server. The MLflow integration itself is tested
via mock assertions rather than a live server.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.models.trainer import ModelMetadata, ModelRegistry, ModelTrainer


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def xy() -> tuple[pd.DataFrame, pd.Series]:
    """Tiny dataset: 5 features, 120 samples, linear target."""
    rng = np.random.default_rng(42)
    n = 120
    X = pd.DataFrame(
        {f"f{i}": rng.normal(size=n) for i in range(5)},
    )
    y = pd.Series(
        X["f0"] * 2.0 - X["f1"] * 1.5 + rng.normal(scale=0.05, size=n),
        name="target",
    )
    return X, y


@pytest.fixture()
def fast_trainer() -> ModelTrainer:
    """ModelTrainer configured for speed: 2 trials, linear models only."""
    return ModelTrainer(
        n_trials=2,
        cv_folds=2,
        n_jobs=1,
        random_seed=0,
        search_space=["linear"],
        mlflow_cfg=None,   # disable tracking in unit tests
    )


# ─── ModelMetadata ────────────────────────────────────────────────────────────


class TestModelMetadata:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        meta = ModelMetadata(
            model_type="lightgbm",
            best_params={"n_estimators": 400, "learning_rate": 0.05},
            cv_rmse=0.8421,
            n_trials=50,
            n_features=15,
            feature_names=["f0", "f1", "f2"],
            train_rows=17000,
        )
        path = tmp_path / "metadata.json"
        meta.save(path)
        loaded = ModelMetadata.load(path)
        assert loaded.model_type == "lightgbm"
        assert loaded.cv_rmse == pytest.approx(0.8421)
        assert loaded.feature_names == ["f0", "f1", "f2"]

    def test_json_is_human_readable(self, tmp_path: Path) -> None:
        meta = ModelMetadata(
            model_type="xgboost",
            best_params={},
            cv_rmse=1.0,
            n_trials=10,
            n_features=5,
            feature_names=[],
            train_rows=100,
        )
        path = tmp_path / "meta.json"
        meta.save(path)
        raw = json.loads(path.read_text())
        assert "model_type" in raw
        assert "cv_rmse" in raw


# ─── ModelTrainer ─────────────────────────────────────────────────────────────


class TestModelTrainer:
    def test_fit_produces_best_model(
        self,
        fast_trainer: ModelTrainer,
        xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = xy
        fast_trainer.fit(X, y)
        assert fast_trainer.best_model is not None

    def test_fit_populates_metadata(
        self,
        fast_trainer: ModelTrainer,
        xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = xy
        fast_trainer.fit(X, y)
        meta = fast_trainer.metadata
        assert meta.n_features == X.shape[1]
        assert meta.train_rows == len(X)
        assert meta.cv_rmse > 0
        assert meta.model_type in ("linear",)   # only linear in search_space

    def test_fit_produces_optuna_study(
        self,
        fast_trainer: ModelTrainer,
        xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = xy
        fast_trainer.fit(X, y)
        assert len(fast_trainer.study.trials) == 2

    def test_best_model_can_predict(
        self,
        fast_trainer: ModelTrainer,
        xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = xy
        fast_trainer.fit(X, y)
        preds = fast_trainer.best_model.predict(X)
        assert preds.shape == (len(X),)

    def test_refit_on_top_k_changes_feature_count(
        self,
        fast_trainer: ModelTrainer,
        xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = xy
        fast_trainer.fit(X, y)
        top3 = X.columns[:3].tolist()
        fast_trainer.refit_on_top_k(X, y, top3)
        assert fast_trainer.metadata.n_features == 3
        assert fast_trainer.metadata.feature_names == top3

    def test_refit_requires_fit_first(self, fast_trainer: ModelTrainer) -> None:
        with pytest.raises(RuntimeError, match="Call fit()"):
            fast_trainer.refit_on_top_k(
                pd.DataFrame({"a": [1.0]}),
                pd.Series([1.0]),
                ["a"],
            )

    def test_accessing_model_before_fit_raises(self, fast_trainer: ModelTrainer) -> None:
        with pytest.raises(RuntimeError):
            _ = fast_trainer.best_model

    def test_accessing_study_before_fit_raises(self, fast_trainer: ModelTrainer) -> None:
        with pytest.raises(RuntimeError):
            _ = fast_trainer.study

    def test_mlflow_run_id_none_when_disabled(
        self,
        fast_trainer: ModelTrainer,
        xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = xy
        fast_trainer.fit(X, y)
        assert fast_trainer.mlflow_run_id is None

    def test_returns_self_for_chaining(
        self,
        fast_trainer: ModelTrainer,
        xy: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = xy
        result = fast_trainer.fit(X, y)
        assert result is fast_trainer


# ─── MLflow integration (mocked) ──────────────────────────────────────────────


class TestMLflowIntegration:
    """
    Verify that ModelTrainer calls the correct MLflow functions
    when tracking is enabled, without requiring a live MLflow server.
    """

    def test_mlflow_start_run_called_when_tracking_enabled(
        self, xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = xy
        trainer = ModelTrainer(
            n_trials=2,
            cv_folds=2,
            n_jobs=1,
            search_space=["linear"],
            mlflow_cfg={"auto_log_params": True, "log_model": False},
        )
        with patch("mlflow.start_run") as mock_run, \
             patch("mlflow.log_params"), \
             patch("mlflow.log_metric"), \
             patch("mlflow.log_metrics"), \
             patch("mlflow.set_tags"), \
             patch("mlflow.set_tag"):
            # Make the context manager return a mock run object
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_ctx.info.run_id = "test-run-id-123"
            mock_run.return_value = mock_ctx

            trainer.fit(X, y)
            mock_run.assert_called_once()

    def test_promote_if_better_skips_when_no_run_id(self) -> None:
        trainer = ModelTrainer(
            n_trials=2, cv_folds=2, n_jobs=1,
            search_space=["linear"], mlflow_cfg={}
        )
        # Should not raise — just silently skip
        trainer.promote_if_better(cv_rmse=0.5)

    def test_log_eval_metrics_skips_when_no_run_id(self) -> None:
        trainer = ModelTrainer(n_trials=2, cv_folds=2, n_jobs=1)
        # Should not raise
        trainer.log_eval_metrics(rmse=0.5, mae=0.4, r2=0.9, top_k=15)


# ─── ModelRegistry ────────────────────────────────────────────────────────────


class TestModelRegistry:
    def _make_fitted_trainer(
        self, xy: tuple[pd.DataFrame, pd.Series]
    ) -> ModelTrainer:
        X, y = xy
        trainer = ModelTrainer(
            n_trials=2, cv_folds=2, n_jobs=1,
            search_space=["linear"], mlflow_cfg=None,
        )
        trainer.fit(X, y)
        return trainer

    def test_save_creates_expected_files(
        self, tmp_path: Path, xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        trainer = self._make_fitted_trainer(xy)
        registry = ModelRegistry(model_dir=tmp_path)
        registry.save(trainer, top_features=["f0", "f1"], filename="model.pkl")

        assert (tmp_path / "model.pkl").exists()
        assert (tmp_path / "optuna_study.pkl").exists()
        assert (tmp_path / "model_metadata.json").exists()
        assert (tmp_path / "top_features.json").exists()

    def test_load_model_returns_estimator(
        self, tmp_path: Path, xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        trainer = self._make_fitted_trainer(xy)
        registry = ModelRegistry(model_dir=tmp_path)
        registry.save(trainer, filename="model.pkl")
        loaded = registry.load_model("model.pkl")
        X, _ = xy
        preds = loaded.predict(X)
        assert len(preds) == len(X)

    def test_load_top_features(
        self, tmp_path: Path, xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        trainer = self._make_fitted_trainer(xy)
        registry = ModelRegistry(model_dir=tmp_path)
        features = ["f0", "f2", "f4"]
        registry.save(trainer, top_features=features)
        loaded = registry.load_top_features()
        assert loaded == features

    def test_load_metadata(
        self, tmp_path: Path, xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        trainer = self._make_fitted_trainer(xy)
        registry = ModelRegistry(model_dir=tmp_path)
        registry.save(trainer)
        meta = registry.load_metadata()
        assert isinstance(meta, ModelMetadata)
        assert meta.model_type in ("linear", "lightgbm", "xgboost", "random_forest")

    def test_load_model_missing_raises(self, tmp_path: Path) -> None:
        registry = ModelRegistry(model_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            registry.load_model("nonexistent.pkl")

    def test_load_top_features_missing_raises(self, tmp_path: Path) -> None:
        registry = ModelRegistry(model_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            registry.load_top_features()

    def test_save_uploads_to_storage_backend(
        self, tmp_path: Path, xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """When a StorageBackend is provided, save() should call upload() for each artifact."""
        trainer = self._make_fitted_trainer(xy)
        mock_storage = MagicMock()
        registry = ModelRegistry(model_dir=tmp_path, storage=mock_storage)
        registry.save(trainer, top_features=["f0"], filename="model.pkl")

        upload_calls = [c[0][1] for c in mock_storage.upload.call_args_list]
        assert any("model.pkl" in k for k in upload_calls)
        assert any("model_metadata.json" in k for k in upload_calls)
