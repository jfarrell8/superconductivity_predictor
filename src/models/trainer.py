"""
src/models/trainer.py
──────────────────────
Handles hyperparameter optimisation (Optuna TPE), model fitting, MLflow
experiment tracking, and artifact persistence (joblib + JSON metadata).

Design
──────
• ModelTrainer owns the Optuna study and the best fitted estimator.
• MLflow tracking is opt-in via mlflow_cfg; the trainer works identically
  with mlflow_cfg=None (useful in unit tests and local dev without a server).
• ModelRegistry provides save/load for both local and S3 storage.
• All hyperparameter search-space logic is in _build_model(), keeping
  the objective function clean and testable.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

# Suppress verbose Optuna / LightGBM output in production
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─── Registry metadata ────────────────────────────────────────────────────────


@dataclass
class ModelMetadata:
    """Serialisable record of a trained model's provenance and performance."""

    model_type: str
    best_params: dict[str, Any]
    cv_rmse: float
    n_trials: int
    n_features: int
    feature_names: list[str]
    train_rows: int

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ModelMetadata":
        return cls(**json.loads(path.read_text()))


# ─── Model builder ────────────────────────────────────────────────────────────


def _build_model(trial: optuna.Trial, search_space: list[str]) -> BaseEstimator:
    """
    Suggest hyperparameters for a trial and return an unfitted estimator.
    Separating this from the objective keeps each concern testable.
    """
    model_type = trial.suggest_categorical("model_type", search_space)

    if model_type == "linear":
        reg_type = trial.suggest_categorical("reg_type", ["ridge", "lasso", "elasticnet"])
        alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
        if reg_type == "ridge":
            return Ridge(alpha=alpha)
        if reg_type == "lasso":
            return Lasso(alpha=alpha, max_iter=5000)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)

    if model_type == "lightgbm":
        return LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 10, 200),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            random_state=42,
            verbose=-1,
        )

    if model_type == "xgboost":
        return XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 0, 5),
            random_state=42,
            verbosity=0,
        )

    # random_forest
    return RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
        random_state=42,
        n_jobs=-1,
    )


# ─── MLflow helpers ───────────────────────────────────────────────────────────


def _configure_mlflow(cfg: dict) -> Optional[str]:
    """
    Point MLflow at the tracking server from config and return the
    experiment ID (creating the experiment if it does not yet exist).
    Returns None if MLflow config is absent (disables tracking).
    """
    import mlflow

    exp_cfg = cfg.get("experiment", {})
    tracking_uri = exp_cfg.get("mlflow_tracking_uri")
    experiment_name = exp_cfg.get("mlflow_experiment_name", "default")

    if not tracking_uri:
        return None

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"MLflow: created experiment '{experiment_name}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id  # type: ignore[return-value]


def _promote_model_if_better(
    run_id: str,
    cv_rmse: float,
    registered_model_name: str,
    threshold: float,
) -> None:
    """
    Register the model from this run with the MLflow Model Registry and
    promote it to Staging if its CV RMSE beats the threshold.

    Lifecycle: None → Staging (auto) → Production (manual approval gate).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    # Register (creates the registered model if it doesn't exist yet)
    mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    logger.info(
        f"MLflow: registered model '{registered_model_name}' "
        f"version {mv.version} from run {run_id}"
    )

    if cv_rmse < threshold:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False,
        )
        logger.info(
            f"MLflow: promoted v{mv.version} to Staging "
            f"(CV RMSE={cv_rmse:.4f} < threshold={threshold})"
        )
    else:
        logger.info(
            f"MLflow: model NOT promoted — CV RMSE={cv_rmse:.4f} "
            f">= threshold={threshold}"
        )


# ─── Trainer ──────────────────────────────────────────────────────────────────


class ModelTrainer:
    """
    Runs an Optuna hyperparameter optimisation study and fits the best model,
    with optional MLflow experiment tracking throughout.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials.
    cv_folds : int
        Number of cross-validation folds used to score each trial.
    n_jobs : int
        Parallelism passed to Optuna. -1 = use all cores.
    random_seed : int
        Controls reproducibility of the TPE sampler and CV splits.
    search_space : list[str]
        Which model families to include.
    mlflow_cfg : dict | None
        Parsed mlflow section of experiment_v1.yaml. Pass None to disable
        tracking (default for unit tests).
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        n_jobs: int = -1,
        random_seed: int = 42,
        search_space: Optional[list[str]] = None,
        mlflow_cfg: Optional[dict] = None,
    ) -> None:
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.search_space = search_space or [
            "linear", "lightgbm", "xgboost", "random_forest"
        ]
        self.mlflow_cfg = mlflow_cfg or {}
        self._study: Optional[optuna.Study] = None
        self._best_model: Optional[BaseEstimator] = None
        self._metadata: Optional[ModelMetadata] = None
        self._mlflow_run_id: Optional[str] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        extra_tags: Optional[dict[str, str]] = None,
    ) -> "ModelTrainer":
        """
        Run HPO and fit the winning model on the full training set.

        If mlflow_cfg is set, opens a run that logs:
          - Every Optuna trial as a nested child run (params + RMSE)
          - Best params, final CV RMSE, and test metrics as the parent run
          - The fitted model artifact via mlflow.sklearn/xgboost/lightgbm
          - Tags: model_type, n_trials, n_features, git SHA (if available)
        """
        import mlflow

        tracking_enabled = bool(self.mlflow_cfg.get("auto_log_params"))
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)

        with (
            mlflow.start_run(tags=extra_tags or {}) if tracking_enabled
            else _null_context()
        ) as run:

            if tracking_enabled and run is not None:
                self._mlflow_run_id = run.info.run_id
                mlflow.set_tags({
                    "n_trials": str(self.n_trials),
                    "cv_folds": str(self.cv_folds),
                    "random_seed": str(self.random_seed),
                    "search_space": ",".join(self.search_space),
                })

            def objective(trial: optuna.Trial) -> float:
                model = _build_model(trial, self.search_space)
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring="neg_mean_squared_error",
                )
                rmse = float(np.sqrt(-scores.mean()))

                # Log each trial as a nested MLflow run
                if tracking_enabled:
                    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                        mlflow.log_params(trial.params)
                        mlflow.log_metric("cv_rmse", rmse)

                return rmse

            self._study = optuna.create_study(
                direction="minimize",
                sampler=TPESampler(seed=self.random_seed),
                study_name="superconductivity_hpo",
            )
            self._study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

            best_rmse = self._study.best_value
            best_params = self._study.best_params
            logger.info(
                f"HPO complete. Best CV RMSE={best_rmse:.4f} "
                f"with params={best_params}"
            )

            # ── Rebuild and fit the winner on the full training set ──────────
            best_trial = self._study.best_trial
            self._best_model = _build_model(best_trial, self.search_space)  # type: ignore[arg-type]
            self._best_model.fit(X_train, y_train)

            self._metadata = ModelMetadata(
                model_type=best_params.get("model_type", "unknown"),
                best_params=best_params,
                cv_rmse=round(best_rmse, 6),
                n_trials=self.n_trials,
                n_features=X_train.shape[1],
                feature_names=X_train.columns.tolist(),
                train_rows=len(X_train),
            )

            # ── MLflow: log best-run params, metrics, and model artifact ────
            if tracking_enabled:
                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_rmse", best_rmse)
                mlflow.set_tag("model_type", self._metadata.model_type)
                mlflow.set_tag("n_features", str(X_train.shape[1]))

                if self.mlflow_cfg.get("log_model", True):
                    self._log_model_artifact()

        return self

    def log_eval_metrics(
        self,
        rmse: float,
        mae: float,
        r2: float,
        top_k: int,
    ) -> None:
        """
        Log test-set metrics back to the active MLflow run.
        Called by the orchestration flow after evaluation.
        """
        import mlflow
        if self._mlflow_run_id is None:
            return
        with mlflow.start_run(run_id=self._mlflow_run_id):
            mlflow.log_metrics({
                f"test_rmse_top{top_k}": rmse,
                f"test_mae_top{top_k}": mae,
                f"test_r2_top{top_k}": r2,
            })

    def promote_if_better(self, cv_rmse: float) -> None:
        """
        Promote the MLflow model version to Staging if CV RMSE beats
        the configured threshold.
        """
        if not self._mlflow_run_id:
            return
        registered_name = self.mlflow_cfg.get(
            "registered_model_name", "superconductivity-tc-predictor"
        )
        threshold = float(self.mlflow_cfg.get("staging_rmse_threshold", 1.0))
        if self.mlflow_cfg.get("promote_to_staging_on_improvement", True):
            _promote_model_if_better(
                run_id=self._mlflow_run_id,
                cv_rmse=cv_rmse,
                registered_model_name=registered_name,
                threshold=threshold,
            )

    def refit_on_top_k(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        top_features: list[str],
    ) -> "ModelTrainer":
        """Refit the best model architecture on a reduced feature set."""
        if self._best_model is None:
            raise RuntimeError("Call fit() before refit_on_top_k().")
        self._best_model.fit(X_train[top_features], y_train)
        if self._metadata:
            self._metadata.feature_names = top_features
            self._metadata.n_features = len(top_features)
        logger.info(f"Model refit on top {len(top_features)} features.")
        return self

    @property
    def best_model(self) -> BaseEstimator:
        if self._best_model is None:
            raise RuntimeError("Call fit() first.")
        return self._best_model

    @property
    def study(self) -> optuna.Study:
        if self._study is None:
            raise RuntimeError("Call fit() first.")
        return self._study

    @property
    def metadata(self) -> ModelMetadata:
        if self._metadata is None:
            raise RuntimeError("Call fit() first.")
        return self._metadata

    @property
    def mlflow_run_id(self) -> Optional[str]:
        return self._mlflow_run_id

    # ── Private helpers ──────────────────────────────────────────────────────
    
    def _log_model_artifact(self) -> None:
        """Log the fitted model + all training visualizations to MLflow."""
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — safe for scripts
        import matplotlib.pyplot as plt
        import mlflow

        model_type = self._metadata.model_type if self._metadata else "unknown"

        # ── 1. Log the model artifact ────────────────────────────────────────
        try:
            if model_type == "xgboost":
                import mlflow.xgboost
                mlflow.xgboost.log_model(self._best_model, artifact_path="model")
            elif model_type == "lightgbm":
                import mlflow.lightgbm
                mlflow.lightgbm.log_model(self._best_model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(self._best_model, artifact_path="model")
            logger.info(f"MLflow: logged {model_type} model artifact")
        except Exception as exc:
            logger.warning(f"MLflow model logging failed (non-fatal): {exc}")

        # ── 2. Per-boosting-round learning curve ─────────────────────────────
        try:
            if model_type == "xgboost" and hasattr(self._best_model, "evals_result_"):
                results = self._best_model.evals_result_
                for dataset, metrics in results.items():
                    for metric, values in metrics.items():
                        for step, val in enumerate(values):
                            mlflow.log_metric(
                                f"{dataset}_{metric}", val, step=step
                            )
                logger.info("MLflow: logged XGBoost learning curve")

            elif model_type == "lightgbm":
                # LightGBM exposes best_score_ per metric
                if hasattr(self._best_model, "best_score_"):
                    for dataset, metrics in self._best_model.best_score_.items():
                        for metric, value in metrics.items():
                            mlflow.log_metric(f"lgbm_{dataset}_{metric}", value)
                logger.info("MLflow: logged LightGBM best scores")

        except Exception as exc:
            logger.warning(f"Learning curve logging failed (non-fatal): {exc}")

        # ── 3. Optuna optimization history plot ──────────────────────────────
        try:
            if self._study is not None:
                trials = self._study.trials
                best_so_far = []
                current_best = float("inf")
                for t in trials:
                    if t.value is not None and t.value < current_best:
                        current_best = t.value
                    best_so_far.append(current_best)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(
                    range(1, len(best_so_far) + 1),
                    best_so_far,
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                    color="#1D9E75",
                )
                ax.set_xlabel("Trial number")
                ax.set_ylabel("Best CV RMSE")
                ax.set_title("Optuna optimisation history")
                ax.grid(True, linestyle="--", alpha=0.4)
                fig.tight_layout()
                mlflow.log_figure(fig, "plots/optuna_history.png")
                plt.close(fig)
                logger.info("MLflow: logged Optuna optimisation history plot")
        except Exception as exc:
            logger.warning(f"Optuna history plot failed (non-fatal): {exc}")


# ─── Registry ─────────────────────────────────────────────────────────────────


class ModelRegistry:
    """
    Persists and loads trained artifacts to/from a local directory or S3.

    Artifacts saved
    ───────────────
    • best_model.pkl       — fitted sklearn-compatible estimator
    • optuna_study.pkl     — full Optuna study (for reproducibility)
    • model_metadata.json  — human-readable provenance record
    • top_features.json    — ordered list of selected feature names
    """

    def __init__(
        self,
        model_dir: Path | str,
        storage: Optional[Any] = None,  # StorageBackend, typed loosely to avoid circular import
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._storage = storage  # None = local only

    def save(
        self,
        trainer: ModelTrainer,
        top_features: Optional[list[str]] = None,
        filename: str = "best_model.pkl",
    ) -> None:
        # ── Write locally first ──────────────────────────────────────────────
        local_model = self.model_dir / filename
        joblib.dump(trainer.best_model, local_model)
        joblib.dump(trainer.study, self.model_dir / "optuna_study.pkl")
        trainer.metadata.save(self.model_dir / "model_metadata.json")
        if top_features:
            (self.model_dir / "top_features.json").write_text(
                json.dumps(top_features, indent=2)
            )
        logger.info(f"Artifacts saved locally to {self.model_dir}")

        # ── Mirror to S3 if a storage backend is configured ─────────────────
        if self._storage is not None:
            for fname in [filename, "optuna_study.pkl", "model_metadata.json", "top_features.json"]:
                local_path = self.model_dir / fname
                if local_path.exists():
                    self._storage.upload(local_path, f"models/{fname}")
            logger.info("Artifacts mirrored to remote storage.")

    def load_model(self, filename: str = "best_model.pkl") -> BaseEstimator:
        path = self.model_dir / filename
        # Pull from S3 if not present locally
        if not path.exists() and self._storage is not None:
            logger.info(f"Model not found locally — downloading from remote storage.")
            self._storage.download(f"models/{filename}", path)
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model

    def load_top_features(self) -> list[str]:
        path = self.model_dir / "top_features.json"
        if not path.exists() and self._storage is not None:
            self._storage.download("models/top_features.json", path)
        if not path.exists():
            raise FileNotFoundError(f"top_features.json not found in {self.model_dir}")
        return json.loads(path.read_text())

    def load_metadata(self) -> ModelMetadata:
        path = self.model_dir / "model_metadata.json"
        if not path.exists() and self._storage is not None:
            self._storage.download("models/model_metadata.json", path)
        return ModelMetadata.load(path)


# ─── Null context manager (MLflow tracking disabled) ─────────────────────────


class _NullRun:
    """Mimics mlflow.ActiveRun enough to satisfy the `with` statement."""
    info = type("info", (), {"run_id": None})()
    def __enter__(self) -> "_NullRun": return self
    def __exit__(self, *_: Any) -> None: pass


def _null_context() -> _NullRun:
    return _NullRun()
