"""
src/api/sagemaker/inference.py
───────────────────────────────
SageMaker entry-point script for the built-in scikit-learn container.

SageMaker calls these four functions in this order:
  model_fn      — load the model artifact from /opt/ml/model
  input_fn      — deserialise the raw request body into a DataFrame
  predict_fn    — run inference
  output_fn     — serialise the prediction to the response format

The model artifact (best_model_top15.pkl + top_features.json) is packaged
into a model.tar.gz by the CI/CD pipeline and uploaded to S3 before
SageMaker pulls it at endpoint creation time.

Packaging (run from project root after training):
    mkdir -p /tmp/sm_model
    cp models/best_model_top15.pkl /tmp/sm_model/
    cp models/top_features.json /tmp/sm_model/
    cp src/api/sagemaker/inference.py /tmp/sm_model/
    tar -czf models/best_model_top15.tar.gz -C /tmp/sm_model .
    aws s3 cp models/best_model_top15.tar.gz s3://your-bucket/models/
"""

from __future__ import annotations

import json
import os
from io import StringIO
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ─── Model loading ────────────────────────────────────────────────────────────


def model_fn(model_dir: str) -> dict[str, Any]:
    """
    Load the model and feature list from the SageMaker model directory.
    Called once at container startup; the return value is passed to predict_fn.
    """
    model_path = os.path.join(model_dir, "best_model_top15.pkl")
    features_path = os.path.join(model_dir, "top_features.json")

    model = joblib.load(model_path)
    with open(features_path) as f:
        top_features = json.load(f)

    return {"model": model, "top_features": top_features}


# ─── Input deserialisation ────────────────────────────────────────────────────


def input_fn(request_body: str | bytes, content_type: str) -> pd.DataFrame:
    """
    Deserialise the incoming request into a DataFrame.
    Supports: application/json, text/csv.
    """
    if content_type == "application/json":
        data = json.loads(request_body)
        # Accept both {"features": {...}} (single) and {"instances": [...]} (batch)
        if "features" in data:
            return pd.DataFrame([data["features"]])
        if "instances" in data:
            return pd.DataFrame(data["instances"])
        # Bare dict: assume it's a single feature map
        return pd.DataFrame([data])

    if content_type == "text/csv":
        return pd.read_csv(
            StringIO(request_body if isinstance(request_body, str) else request_body.decode())
        )

    raise ValueError(
        f"Unsupported content type: {content_type}. Use 'application/json' or 'text/csv'."
    )


# ─── Prediction ───────────────────────────────────────────────────────────────


def predict_fn(input_data: pd.DataFrame, model_artifacts: dict[str, Any]) -> np.ndarray:
    """
    Run inference. Selects only the top features the model was trained on,
    guarding against extra columns sent by the caller.
    """
    model = model_artifacts["model"]
    top_features = model_artifacts["top_features"]

    missing = set(top_features) - set(input_data.columns)
    if missing:
        raise ValueError(f"Input is missing required features: {sorted(missing)}")

    return model.predict(input_data[top_features])


# ─── Output serialisation ─────────────────────────────────────────────────────


def output_fn(prediction: np.ndarray, accept: str) -> tuple[str, str]:
    """
    Serialise predictions to the response body.
    Returns (body, content_type).
    """
    if accept in ("application/json", "*/*"):
        body = json.dumps({"predictions": prediction.tolist()})
        return body, "application/json"

    if accept == "text/csv":
        body = "\n".join(str(p) for p in prediction.tolist())
        return body, "text/csv"

    raise ValueError(f"Unsupported accept type: {accept}")
