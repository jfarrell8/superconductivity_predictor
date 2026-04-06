"""
src/monitoring/logger.py
─────────────────────────
Structured prediction logger.

Every inference request is logged as a JSON record containing:
  • timestamp (ISO 8601 UTC)
  • request_id (UUID, generated per request)
  • input features (the 15-feature vector sent to the model)
  • raw prediction (Box-Cox scale)
  • model_type + n_features metadata

These logs serve three purposes:
  1. Debugging — trace any individual prediction back to its exact inputs
  2. Drift monitoring — batch the logs into windows and run DriftMonitor
     against them (same KS-test, different trigger cadence)
  3. SageMaker compatibility — the schema mirrors what SageMaker DataCapture
     produces, so the same downstream analysis works for both local and
     cloud deployments

Usage
─────
    logger = PredictionLogger(sink="logs/predictions.jsonl")
    logger.log(request_id="abc", features={"mean_atomic_mass": 88.5, ...}, prediction=4.31)

    # Load for drift analysis
    df = PredictionLogger.load_as_dataframe("logs/predictions.jsonl")
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger


class PredictionLogger:
    """
    Appends one JSON record per prediction to a JSONL sink file.

    The sink can be a local path (default) or an S3 URI — the caller
    is responsible for routing (i.e. passing an open file-like object
    backed by s3fs when running in production).

    Parameters
    ----------
    sink : str | Path
        Path to the JSONL log file. Created if it does not exist.
    max_file_mb : float
        Rotate when the sink exceeds this size (default 100 MB).
        A new file is created with a timestamp suffix.
    """

    def __init__(
        self,
        sink: str | Path = "logs/predictions.jsonl",
        max_file_mb: float = 100.0,
    ) -> None:
        self.sink = Path(sink)
        self.max_file_mb = max_file_mb
        self.sink.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def log(
        self,
        features: dict[str, float],
        prediction: float,
        request_id: str | None = None,
        model_type: str = "unknown",
        n_features: int = 0,
    ) -> str:
        """
        Append one prediction record to the JSONL sink.

        Parameters
        ----------
        features    : feature name → value dict (the model input)
        prediction  : raw model output (Box-Cox scale)
        request_id  : caller-supplied ID for tracing; auto-generated if None
        model_type  : e.g. "lightgbm"
        n_features  : number of features used

        Returns
        -------
        str : the request_id used (useful when auto-generated)
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model_type": model_type,
            "n_features": n_features,
            "prediction": prediction,
            "features": features,
        }

        self._rotate_if_needed()
        with self.sink.open("a") as f:
            f.write(json.dumps(record) + "\n")

        return request_id

    def log_batch(
        self,
        feature_rows: list[dict[str, float]],
        predictions: list[float],
        model_type: str = "unknown",
        n_features: int = 0,
    ) -> list[str]:
        """Log a batch of predictions, returning the list of request IDs."""
        return [
            self.log(
                features=feat,
                prediction=pred,
                model_type=model_type,
                n_features=n_features,
            )
            for feat, pred in zip(feature_rows, predictions, strict=False)
        ]

    @staticmethod
    def load_as_dataframe(sink: str | Path) -> Any:
        """
        Load a JSONL prediction log into a pandas DataFrame.

        Features are expanded into top-level columns so the DataFrame
        can be passed directly to DriftMonitor.detect().

        Returns
        -------
        pd.DataFrame with columns: timestamp, request_id, model_type,
        n_features, prediction, + one column per feature.
        """
        import pandas as pd

        records = []
        with open(sink) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    flat = {k: v for k, v in record.items() if k != "features"}
                    flat.update(record.get("features", {}))
                    records.append(flat)

        return pd.DataFrame(records)

    # ── Private ───────────────────────────────────────────────────────────────

    def _rotate_if_needed(self) -> None:
        """If sink exceeds max_file_mb, rename it and start a fresh file."""
        if not self.sink.exists():
            return
        size_mb = self.sink.stat().st_size / (1024 * 1024)
        if size_mb >= self.max_file_mb:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            rotated = self.sink.with_name(f"{self.sink.stem}_{ts}{self.sink.suffix}")
            self.sink.rename(rotated)
            _loguru_logger.info(f"Prediction log rotated: {rotated} ({size_mb:.1f} MB)")
