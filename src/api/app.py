"""
src/api/app.py
───────────────
FastAPI REST service exposing model inference, health checks, drift
monitoring, and prediction logging.

Endpoints
─────────
GET  /health                  — liveness probe
GET  /model/info              — metadata about the loaded model
POST /predict                 — single-sample prediction (logged)
POST /predict/batch           — batch predictions (logged)
GET  /monitoring/drift        — KS-test drift detection vs. reference data
GET  /monitoring/logs         — summary statistics from the prediction log
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from src.models.trainer import ModelMetadata, ModelRegistry
from src.monitoring.drift import DriftMonitor
from src.monitoring.logger import PredictionLogger


# ─── Application settings ─────────────────────────────────────────────────────


class Settings(BaseModel):
    model_dir: str = "models"
    model_filename: str = "best_model_top15.pkl"
    reference_data_path: str = "data/processed/train_reduced.csv"
    drift_significance_level: float = 0.05
    prediction_log_path: str = "logs/predictions.jsonl"
    prediction_log_max_mb: float = 100.0


settings = Settings()


# ─── App state ────────────────────────────────────────────────────────────────


class AppState:
    model: Any = None
    metadata: ModelMetadata | None = None
    top_features: list[str] = []
    drift_monitor: DriftMonitor | None = None
    prediction_logger: PredictionLogger | None = None


app_state = AppState()


# ─── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load all artifacts once at startup; clean up at shutdown."""
    logger.info("Loading model artifacts…")
    registry = ModelRegistry(settings.model_dir)

    try:
        app_state.model = registry.load_model(settings.model_filename)
        app_state.metadata = registry.load_metadata()
        app_state.top_features = registry.load_top_features()
        logger.info(
            f"Model loaded: {app_state.metadata.model_type} | "
            f"{app_state.metadata.n_features} features"
        )
    except FileNotFoundError as exc:
        logger.error(f"Artifact not found: {exc}")
        raise RuntimeError(
            "Model artifacts missing — run the training pipeline first."
        ) from exc

    # Initialise drift monitor against training reference data
    if Path(settings.reference_data_path).exists():
        ref_df = pd.read_csv(settings.reference_data_path)
        ref_X = ref_df[app_state.top_features] if app_state.top_features else ref_df
        app_state.drift_monitor = DriftMonitor(
            reference_data=ref_X,
            significance_level=settings.drift_significance_level,
        )
        logger.info("Drift monitor initialised.")
    else:
        logger.warning("Reference data not found — drift monitoring disabled.")

    # Initialise prediction logger
    app_state.prediction_logger = PredictionLogger(
        sink=settings.prediction_log_path,
        max_file_mb=settings.prediction_log_max_mb,
    )
    logger.info(f"Prediction logger initialised → {settings.prediction_log_path}")

    yield  # ← app is running

    logger.info("Shutting down.")


# ─── FastAPI app ───────────────────────────────────────────────────────────────


app = FastAPI(
    title="Superconductivity Critical Temperature Predictor",
    description=(
        "Predicts the critical temperature (Tc) of superconducting materials "
        "from compositionally-derived physical features. "
        "Built with XGBoost / LightGBM / RandomForest optimised via Optuna."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    """
    Feature values for a single superconductor sample.

    Only the top-15 selected features are required for inference.
    Feature names match the column names in the processed dataset.
    """

    features: dict[str, float] = Field(
        ...,
        example={
            "mean_atomic_mass": 88.5,
            "wtd_mean_atomic_mass": 91.2,
            "gmean_atomic_mass": 85.3,
            "mean_fie": 6.2,
            "mean_atomic_radius": 1.55,
            "wtd_mean_atomic_radius": 1.48,
            "mean_Density": 5400.0,
            "wtd_mean_Density": 5100.0,
            "mean_ElectronAffinity": 1.2,
            "wtd_mean_ElectronAffinity": 1.1,
            "mean_FusionHeat": 14.5,
            "wtd_mean_FusionHeat": 13.8,
            "mean_ThermalConductivity": 12.0,
            "wtd_mean_ThermalConductivity": 11.5,
            "num_elements_simplified": 3,
        },
    )

    @model_validator(mode="after")
    def check_required_features(self) -> "PredictionRequest":
        if app_state.top_features:
            missing = set(app_state.top_features) - set(self.features.keys())
            if missing:
                raise ValueError(f"Missing required features: {sorted(missing)}")
        return self


class PredictionResponse(BaseModel):
    predicted_critical_temp_boxcox: float = Field(
        ..., description="Prediction in Box-Cox transformed scale"
    )
    request_id: str = Field(..., description="Unique ID for tracing this prediction")
    model_type: str
    n_features_used: int


class BatchPredictionRequest(BaseModel):
    samples: list[dict[str, float]] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    request_ids: list[str]
    n_samples: int
    model_type: str


class ModelInfoResponse(BaseModel):
    model_type: str
    cv_rmse: float
    n_features: int
    feature_names: list[str]
    train_rows: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_features: int
    prediction_logging_enabled: bool


class LogSummaryResponse(BaseModel):
    n_predictions_logged: int
    log_path: str
    prediction_stats: dict[str, float]


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health() -> HealthResponse:
    """Liveness probe — returns 200 if the model is loaded."""
    return HealthResponse(
        status="ok" if app_state.model is not None else "degraded",
        model_loaded=app_state.model is not None,
        n_features=len(app_state.top_features),
        prediction_logging_enabled=app_state.prediction_logger is not None,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """Return provenance and performance metadata for the loaded model."""
    if app_state.metadata is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    m = app_state.metadata
    return ModelInfoResponse(
        model_type=m.model_type,
        cv_rmse=m.cv_rmse,
        n_features=m.n_features,
        feature_names=m.feature_names,
        train_rows=m.train_rows,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict critical temperature for a single superconductor sample.

    Returns the prediction in Box-Cox–transformed scale. Use
    `scipy.special.inv_boxcox(value, lambda)` to convert back to Kelvin.
    Every request is logged to the JSONL prediction log for drift analysis.
    """
    if app_state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    try:
        features = app_state.top_features or list(request.features.keys())
        X = pd.DataFrame([request.features])[features]
        prediction = float(app_state.model.predict(X)[0])
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # Log every prediction for offline drift analysis
    model_type = app_state.metadata.model_type if app_state.metadata else "unknown"
    request_id = "unlogged"
    if app_state.prediction_logger is not None:
        request_id = app_state.prediction_logger.log(
            features=request.features,
            prediction=prediction,
            model_type=model_type,
            n_features=len(features),
        )

    return PredictionResponse(
        predicted_critical_temp_boxcox=prediction,
        request_id=request_id,
        model_type=model_type,
        n_features_used=len(features),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Batch inference endpoint — accepts up to 1,000 samples per request.
    Every sample is logged individually to the prediction log.
    """
    if app_state.model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    try:
        features = app_state.top_features
        X = pd.DataFrame(request.samples)[features]
        predictions = app_state.model.predict(X).tolist()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    model_type = app_state.metadata.model_type if app_state.metadata else "unknown"
    request_ids: list[str] = []
    if app_state.prediction_logger is not None:
        request_ids = app_state.prediction_logger.log_batch(
            feature_rows=request.samples,
            predictions=predictions,
            model_type=model_type,
            n_features=len(features),
        )
    else:
        request_ids = ["unlogged"] * len(predictions)

    return BatchPredictionResponse(
        predictions=predictions,
        request_ids=request_ids,
        n_samples=len(predictions),
        model_type=model_type,
    )


@app.get("/monitoring/drift", tags=["Monitoring"])
async def drift_report(data_path: str | None = None) -> dict[str, Any]:
    """
    Run KS-test drift detection on a provided CSV or the reference training data.

    Query param `data_path` — path to a CSV of new observations (same schema).
    If omitted, runs a self-check against the reference training data.
    """
    if app_state.drift_monitor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drift monitor not initialised — reference data not found.",
        )
    if data_path:
        try:
            new_df = pd.read_csv(data_path)
            new_X = new_df[app_state.top_features]
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
    else:
        new_X = app_state.drift_monitor.reference_data

    report = app_state.drift_monitor.detect(new_X)
    return {"drift_report": report.to_dict(), "drifted_features": report.drifted_features}


@app.get("/monitoring/logs", response_model=LogSummaryResponse, tags=["Monitoring"])
async def log_summary() -> LogSummaryResponse:
    """
    Return summary statistics from the prediction log.

    Loads the JSONL log and computes count + prediction distribution stats.
    Useful for quick sanity checks and ad-hoc monitoring without a full
    drift report.
    """
    log_path = Path(settings.prediction_log_path)
    if not log_path.exists():
        return LogSummaryResponse(
            n_predictions_logged=0,
            log_path=str(log_path),
            prediction_stats={},
        )
    try:
        df = PredictionLogger.load_as_dataframe(log_path)
        stats: dict[str, float] = {}
        if "prediction" in df.columns and len(df) > 0:
            s = df["prediction"].describe()
            stats = {
                "mean": round(float(s["mean"]), 4),
                "std": round(float(s["std"]), 4),
                "min": round(float(s["min"]), 4),
                "p25": round(float(s["25%"]), 4),
                "p50": round(float(s["50%"]), 4),
                "p75": round(float(s["75%"]), 4),
                "max": round(float(s["max"]), 4),
            }
        return LogSummaryResponse(
            n_predictions_logged=len(df),
            log_path=str(log_path),
            prediction_stats=stats,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading prediction log: {exc}",
        ) from exc


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
