"""
scripts/run_pipeline.py
────────────────────────
CLI entry point that orchestrates the full training pipeline:
  Data ingestion → Feature engineering → Train/test split →
  HPO (Optuna) → Evaluation → Feature reduction → Artifact persistence

Usage
─────
    python scripts/run_pipeline.py --config configs/experiment_v1.yaml
    python scripts/run_pipeline.py --config configs/experiment_v1.yaml --trials 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split

# Add project root to path so `src` is importable without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import DataLoader
from src.evaluation.evaluator import ModelEvaluator
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelRegistry, ModelTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the superconductivity ML training pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_v1.yaml",
        help="Path to YAML experiment config.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override n_trials from config (useful for quick smoke tests).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg["experiment"]["name"]
    seed = cfg["experiment"]["random_seed"]
    logger.info(f"Starting experiment: {exp_name}")

    # ── 1. Data ingestion ────────────────────────────────────────────────────
    loader = DataLoader(
        train_path=cfg["data"]["raw_train_path"],
        metadata_path=cfg["data"].get("raw_metadata_path"),
    ).load()

    manifest = loader.build_manifest()
    logger.info(
        f"Data loaded: {manifest.row_count:,} rows, "
        f"{manifest.column_count} columns, sha256={manifest.sha256[:12]}…"
    )

    # ── 2. Feature engineering ───────────────────────────────────────────────
    engineer = FeatureEngineer.from_config(cfg)
    X_all, y_all = engineer.fit_transform(loader.train)

    # Save processed data
    processed_path = Path(cfg["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df = X_all.copy()
    processed_df[cfg["feature_engineering"]["engineered_target_column"]] = y_all.values
    processed_df.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path}")

    # ── 3. Train/test split ──────────────────────────────────────────────────
    test_size = cfg["preprocessing"]["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed, shuffle=True
    )
    logger.info(
        f"Split: {len(X_train):,} train / {len(X_test):,} test  "
        f"({int((1-test_size)*100)}% / {int(test_size*100)}%)"
    )

    # ── 4. Hyperparameter optimisation ───────────────────────────────────────
    n_trials = args.trials or cfg["modeling"]["n_trials"]
    trainer = ModelTrainer(
        n_trials=n_trials,
        cv_folds=cfg["modeling"]["cv_folds"],
        n_jobs=cfg["modeling"]["n_jobs"],
        random_seed=seed,
        search_space=cfg["modeling"]["search_space"],
    )
    trainer.fit(X_train, y_train)

    # ── 5. Full-feature evaluation ────────────────────────────────────────────
    evaluator = ModelEvaluator(trainer.best_model)
    metrics_full = evaluator.evaluate(X_test, y_test, label="Full-feature test set")

    # ── 6. Feature reduction sweep ────────────────────────────────────────────
    importance_df = evaluator.feature_importances(X_train.columns.tolist())
    reduction_cfg = cfg["evaluation"]

    reduction_df = evaluator.feature_reduction_sweep(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_counts=reduction_cfg["feature_counts"],
        importance_df=importance_df,
    )
    logger.info(f"\nFeature reduction results:\n{reduction_df.to_string(index=False)}")

    # ── 7. Refit on top-k features ─────────────────────────────────────────────
    k = reduction_cfg["final_top_k_features"]
    top_features = evaluator.top_k_features(importance_df, k)
    trainer.refit_on_top_k(X_train, y_train, top_features)

    metrics_topk = evaluator.evaluate(X_test[top_features], y_test, label=f"Top-{k} test set")

    logger.info(
        f"\nPerformance summary:\n"
        f"  Full features ({X_train.shape[1]}): {metrics_full}\n"
        f"  Top {k} features:               {metrics_topk}"
    )

    # ── 8. Persist artifacts ─────────────────────────────────────────────────
    model_dir = Path(cfg["artifacts"]["model_dir"])
    registry = ModelRegistry(model_dir)
    registry.save(
        trainer,
        top_features=top_features,
        filename=cfg["artifacts"]["top_k_model_filename"],
    )

    # Also save the full-feature model separately
    import joblib
    joblib.dump(
        trainer.best_model,
        model_dir / cfg["artifacts"]["model_filename"],
    )
    manifest.save(model_dir / "data_manifest.json")

    logger.info(f"Pipeline complete. Artifacts in '{model_dir}/'")


if __name__ == "__main__":
    main()
