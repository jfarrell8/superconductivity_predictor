"""
src/orchestration/flows.py
───────────────────────────
Prefect 3 flow definitions for the superconductivity training pipeline.

Each pipeline stage is a @task with its own retry policy, timeout, and
logging. The @flow composes them into a DAG with explicit data dependencies.

Why Prefect over a script?
──────────────────────────
• Automatic retry on transient failures (S3 throttling, network blips)
• Full execution history, logs, and artifacts in the Prefect UI
• Parameterised — run the same flow with different configs without code changes
• Schedulable — deploy with a cron schedule for weekly automated retraining
• Observability — every task emits structured events to the Prefect server

Running locally (no server needed):
    python -m src.orchestration.flows

Deploying to a Prefect Cloud work pool:
    prefect deploy src/orchestration/flows.py:training_pipeline \
        --name superconductivity-weekly \
        --pool default-agent-pool \
        --cron "0 2 * * 0"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
import json

# Support running from project root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact
from prefect.tasks import task_input_hash

from src.data.loader import DataLoader, DataManifest
from src.evaluation.evaluator import ModelEvaluator, RegressionMetrics
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelRegistry, ModelTrainer
from src.storage.store import StorageBackend


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _configure_mlflow_if_enabled(cfg: dict) -> None:
    """Point MLflow at the tracking server defined in config."""
    tracking_uri = cfg.get("experiment", {}).get("mlflow_tracking_uri")
    if not tracking_uri:
        return
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = cfg["experiment"].get("mlflow_experiment_name", "default")
    mlflow.set_experiment(exp_name)
    logger.info(f"MLflow tracking: {tracking_uri} | experiment: {exp_name}")


# ─── Tasks ────────────────────────────────────────────────────────────────────


@task(
    name="ingest-data",
    description="Load raw CSV from local disk or S3, validate schema, build provenance manifest.",
    retries=3,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    cache_expiration=None,
)
def ingest_data(cfg: dict) -> tuple[pd.DataFrame, DataManifest]:
    log = get_run_logger()
    storage = StorageBackend.from_config(cfg)

    # If S3 backend: pull the raw file down before DataLoader reads it
    raw_path = cfg["data"]["raw_train_path"]
    if cfg.get("storage", {}).get("backend") == "s3":
        s3_key = cfg["data"]["s3_raw_train_key"]
        log.info(f"Downloading raw data from S3 key: {s3_key}")
        Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
        storage.download(s3_key, raw_path)

    loader = DataLoader(
        train_path=raw_path,
        metadata_path=cfg["data"].get("raw_metadata_path"),
    ).load()

    manifest = loader.build_manifest()
    log.info(
        f"Ingested {manifest.row_count:,} rows × {manifest.column_count} cols | "
        f"sha256={manifest.sha256[:16]}…"
    )
    return loader.train, manifest


@task(
    name="engineer-features",
    description="Bin quasi-categorical features, Box-Cox target, prune correlations, scale.",
    retries=2,
    retry_delay_seconds=5,
)
def engineer_features(
    raw_df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.Series]:
    log = get_run_logger()
    engineer = FeatureEngineer.from_config(cfg)
    X, y = engineer.fit_transform(raw_df)

    # Persist processed data (local + optional S3 mirror)
    processed_path = Path(cfg["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df = X.copy()
    processed_df[cfg["feature_engineering"]["engineered_target_column"]] = y.values
    processed_df.to_csv(processed_path, index=False)

    storage = StorageBackend.from_config(cfg)
    if cfg.get("storage", {}).get("backend") == "s3":
        storage.upload(processed_path, cfg["data"]["s3_processed_key"])

    log.info(f"Feature engineering complete: {X.shape[1]} features, {len(X):,} samples")
    return X, y


@task(
    name="split-data",
    description="Stratified random train/test split.",
    retries=1,
)
def split_data(
    X: pd.DataFrame, y: pd.Series, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    seed = cfg["experiment"]["random_seed"]
    test_size = cfg["preprocessing"]["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True
    )
    get_run_logger().info(
        f"Split: {len(X_train):,} train / {len(X_test):,} test"
    )
    return X_train, X_test, y_train, y_test


@task(
    name="train-model",
    description="Optuna TPE hyperparameter search across 4 model families + full-set fit.",
    retries=1,
    retry_delay_seconds=30,
    timeout_seconds=7200,  # 2-hour cap for expensive HPO runs
)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict,
    n_trials_override: int | None = None,
) -> ModelTrainer:
    log = get_run_logger()
    n_trials = n_trials_override or cfg["modeling"]["n_trials"]
    log.info(f"Starting HPO: {n_trials} trials across {cfg['modeling']['search_space']}")

    trainer = ModelTrainer(
        n_trials=n_trials,
        cv_folds=cfg["modeling"]["cv_folds"],
        n_jobs=cfg["modeling"]["n_jobs"],
        random_seed=cfg["experiment"]["random_seed"],
        search_space=cfg["modeling"]["search_space"],
        mlflow_cfg=cfg.get("mlflow"),
    )
    trainer.fit(X_train, y_train)
    log.info(f"Best CV RMSE: {trainer.metadata.cv_rmse:.4f} ({trainer.metadata.model_type})")
    return trainer


@task(
    name="evaluate-model",
    description="Compute test-set metrics, feature importances, and reduction sweep.",
    retries=1,
)
def evaluate_model(
    trainer: ModelTrainer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cfg: dict,
) -> tuple[RegressionMetrics, list[str]]:
    log = get_run_logger()
    eval_cfg = cfg["evaluation"]

    evaluator = ModelEvaluator(trainer.best_model)
    metrics_full = evaluator.evaluate(X_test, y_test, label="Full-feature test set")
    log.info(f"Full-feature metrics: {metrics_full}")

    importance_df = evaluator.feature_importances(X_train.columns.tolist())
    reduction_df = evaluator.feature_reduction_sweep(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_counts=eval_cfg["feature_counts"],
        importance_df=importance_df,
    )

    k = eval_cfg["final_top_k_features"]
    top_features = evaluator.top_k_features(importance_df, k)
    trainer.refit_on_top_k(X_train, y_train, top_features)
    metrics_topk = evaluator.evaluate(X_test[top_features], y_test, label=f"Top-{k} test set")
    log.info(f"Top-{k} metrics: {metrics_topk}")

    # Log test metrics back to the MLflow run
    trainer.log_eval_metrics(
        rmse=metrics_topk.rmse,
        mae=metrics_topk.mae,
        r2=metrics_topk.r2,
        top_k=k,
    )

    # Log all visual artifacts to MLflow
    if trainer.mlflow_run_id:
        evaluator.log_artifacts_to_mlflow(
            X_test=X_test,
            y_test=y_test,
            top_features=top_features,
            run_id=trainer.mlflow_run_id,
        )

    # Publish a human-readable Prefect artifact (shows up in the UI)
    _publish_metrics_artifact(
        metrics_full=metrics_full,
        metrics_topk=metrics_topk,
        k=k,
        reduction_df=reduction_df,
        trainer=trainer,
    )

    return metrics_topk, top_features


@task(
    name="persist-artifacts",
    retries=3,
    retry_delay_seconds=15,
)
def persist_artifacts(
    trainer: ModelTrainer,
    manifest: DataManifest,
    top_features: list[str],
    cfg: dict,
) -> None:
    log = get_run_logger()
    storage = StorageBackend.from_config(cfg)
    model_dir = Path(cfg["artifacts"]["model_dir"])
    is_s3 = cfg.get("storage", {}).get("backend") == "s3"

    registry = ModelRegistry(model_dir=model_dir, storage=storage if is_s3 else None)
    registry.save(
        trainer,
        top_features=top_features,
        filename=cfg["artifacts"]["top_k_model_filename"],
    )

    import joblib
    full_model_path = model_dir / cfg["artifacts"]["model_filename"]
    joblib.dump(trainer.best_model, full_model_path)

    # Only mirror to remote storage if actually using S3
    if is_s3:
        storage.upload(full_model_path, f"models/{cfg['artifacts']['model_filename']}")
        log.info("Artifacts mirrored to S3.")

    manifest.save(model_dir / "data_manifest.json")
    log.info(f"All artifacts persisted. model_dir={model_dir}")

    trainer.promote_if_better(cv_rmse=trainer.metadata.cv_rmse)


# ─── Markdown artifact helper ─────────────────────────────────────────────────


def _publish_metrics_artifact(
    metrics_full: RegressionMetrics,
    metrics_topk: RegressionMetrics,
    k: int,
    reduction_df: pd.DataFrame,
    trainer: ModelTrainer,
) -> None:
    """Publish a Markdown summary to the Prefect UI."""
    table_rows = "\n".join(
        f"| {row.n_features} | {row.rmse:.4f} |"
        for row in reduction_df.itertuples()
    )
    md = f"""
## Training Run Summary

| Metric | Full Features | Top {k} Features |
|--------|-------------|-----------|
| RMSE   | {metrics_full.rmse:.4f} | {metrics_topk.rmse:.4f} |
| MAE    | {metrics_full.mae:.4f} | {metrics_topk.mae:.4f} |
| R²     | {metrics_full.r2:.4f} | {metrics_topk.r2:.4f} |

**Best model:** `{trainer.metadata.model_type}`
**CV RMSE:** `{trainer.metadata.cv_rmse:.4f}`
**Trials:** `{trainer.metadata.n_trials}`

### Feature Reduction Sweep

| n_features | RMSE |
|------------|------|
{table_rows}
"""
    create_markdown_artifact(
        markdown=md,
        key="training-summary",
        description="Model performance summary for this training run",
    )


# ─── Flow ─────────────────────────────────────────────────────────────────────


@flow(
    name="superconductivity-training-pipeline",
    description=(
        "End-to-end training pipeline: ingest → engineer → split → "
        "HPO → evaluate → persist. Scheduled weekly."
    ),
    version="0.1.0",
)
def training_pipeline(
    config_path: str = "configs/experiment_v1.yaml",
    n_trials_override: int | None = None,
) -> dict[str, Any]:
    """
    Parameterised Prefect flow for the superconductivity ML pipeline.

    Parameters
    ----------
    config_path : str
        Path to the YAML experiment config.
    n_trials_override : int | None
        Override n_trials from config. Useful for quick smoke-test runs
        (e.g. n_trials_override=5).

    Returns
    -------
    dict
        Summary of the run: model_type, cv_rmse, test_rmse, top_features.
    """
    cfg = _load_config(config_path)
    _configure_mlflow_if_enabled(cfg)

    # ── DAG ──────────────────────────────────────────────────────────────────
    raw_df, manifest = ingest_data(cfg)
    X_all, y_all = engineer_features(raw_df, cfg)
    X_train, X_test, y_train, y_test = split_data(X_all, y_all, cfg)

    # Compute and save training medians for inference-time imputation
    medians = X_train.median().to_dict()
    Path(cfg["artifacts"]["model_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["artifacts"]["model_dir"]) / "feature_medians.json").write_text(
        json.dumps(medians, indent=2)
    )

    trainer = train_model(X_train, y_train, cfg, n_trials_override)
    metrics, top_features = evaluate_model(trainer, X_train, X_test, y_train, y_test, cfg)
    persist_artifacts(trainer, manifest, top_features, cfg)

    return {
        "model_type": trainer.metadata.model_type,
        "cv_rmse": trainer.metadata.cv_rmse,
        "test_rmse": metrics.rmse,
        "test_r2": metrics.r2,
        "top_features": top_features,
        "mlflow_run_id": trainer.mlflow_run_id,
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = training_pipeline(n_trials_override=5)
    print("\nPipeline complete:", result)
