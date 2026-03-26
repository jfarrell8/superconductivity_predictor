"""
tests/integration/test_api.py
──────────────────────────────
Integration tests for the FastAPI inference service.

These tests mock the model registry so they run without real trained
artifacts, making them suitable for CI pipelines on every PR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, app_state
from src.models.trainer import ModelMetadata
from src.monitoring.logger import PredictionLogger


# ─── Fixtures ─────────────────────────────────────────────────────────────────


MOCK_FEATURES = [
    "mean_atomic_mass", "wtd_mean_atomic_mass", "gmean_atomic_mass",
    "mean_fie", "mean_atomic_radius", "wtd_mean_atomic_radius",
    "mean_Density", "wtd_mean_Density", "mean_ElectronAffinity",
    "wtd_mean_ElectronAffinity", "mean_FusionHeat", "wtd_mean_FusionHeat",
    "mean_ThermalConductivity", "wtd_mean_ThermalConductivity",
    "num_elements_simplified",
]

MOCK_METADATA = ModelMetadata(
    model_type="lightgbm",
    best_params={"model_type": "lightgbm", "n_estimators": 400},
    cv_rmse=0.8421,
    n_trials=50,
    n_features=15,
    feature_names=MOCK_FEATURES,
    train_rows=17000,
)


@pytest.fixture(autouse=True)
def mock_app_state(tmp_path: Path) -> Generator[None, None, None]:
    """Inject a mock model and a real (tmp) PredictionLogger into app_state."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([4.5])

    original = {
        "model": app_state.model,
        "metadata": app_state.metadata,
        "top_features": app_state.top_features,
        "drift_monitor": app_state.drift_monitor,
        "prediction_logger": app_state.prediction_logger,
    }

    app_state.model = mock_model
    app_state.metadata = MOCK_METADATA
    app_state.top_features = MOCK_FEATURES
    app_state.drift_monitor = None
    app_state.prediction_logger = PredictionLogger(
        sink=tmp_path / "test_predictions.jsonl"
    )

    yield

    app_state.model = original["model"]
    app_state.metadata = original["metadata"]
    app_state.top_features = original["top_features"]
    app_state.drift_monitor = original["drift_monitor"]
    app_state.prediction_logger = original["prediction_logger"]


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


def _sample_payload() -> dict[str, Any]:
    return {
        "features": {feat: float(i + 1) for i, feat in enumerate(MOCK_FEATURES)}
    }


# ─── Health ───────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_model_loaded_true(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["model_loaded"] is True

    def test_reports_feature_count(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["n_features"] == len(MOCK_FEATURES)

    def test_prediction_logging_enabled(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["prediction_logging_enabled"] is True


# ─── Model info ───────────────────────────────────────────────────────────────


class TestModelInfoEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/model/info").status_code == 200

    def test_model_type_correct(self, client: TestClient) -> None:
        body = client.get("/model/info").json()
        assert body["model_type"] == "lightgbm"

    def test_feature_names_present(self, client: TestClient) -> None:
        body = client.get("/model/info").json()
        assert set(body["feature_names"]) == set(MOCK_FEATURES)


# ─── Predict ──────────────────────────────────────────────────────────────────


class TestPredictEndpoint:
    def test_valid_request_returns_200(self, client: TestClient) -> None:
        assert client.post("/predict", json=_sample_payload()).status_code == 200

    def test_prediction_value_is_float(self, client: TestClient) -> None:
        body = client.post("/predict", json=_sample_payload()).json()
        assert isinstance(body["predicted_critical_temp_boxcox"], float)

    def test_request_id_present(self, client: TestClient) -> None:
        body = client.post("/predict", json=_sample_payload()).json()
        assert "request_id" in body
        assert isinstance(body["request_id"], str)
        assert len(body["request_id"]) > 0

    def test_model_type_in_response(self, client: TestClient) -> None:
        body = client.post("/predict", json=_sample_payload()).json()
        assert body["model_type"] == "lightgbm"

    def test_n_features_used_correct(self, client: TestClient) -> None:
        body = client.post("/predict", json=_sample_payload()).json()
        assert body["n_features_used"] == len(MOCK_FEATURES)

    def test_missing_feature_returns_422(self, client: TestClient) -> None:
        payload = _sample_payload()
        del payload["features"]["mean_atomic_mass"]
        assert client.post("/predict", json=payload).status_code == 422

    def test_prediction_is_logged(self, client: TestClient) -> None:
        """After a successful predict call, the logger should have one record."""
        client.post("/predict", json=_sample_payload())
        assert app_state.prediction_logger is not None
        log_path = app_state.prediction_logger.sink
        records = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == 1
        assert "prediction" in records[0]
        assert "request_id" in records[0]
        assert "features" in records[0]

    def test_model_predict_called_once(self, client: TestClient) -> None:
        client.post("/predict", json=_sample_payload())
        app_state.model.predict.assert_called_once()


# ─── Batch predict ────────────────────────────────────────────────────────────


class TestBatchPredictEndpoint:
    def _batch_payload(self, n: int = 3) -> dict[str, Any]:
        sample = {feat: float(i + 1) for i, feat in enumerate(MOCK_FEATURES)}
        return {"samples": [sample] * n}

    def test_returns_200(self, client: TestClient) -> None:
        app_state.model.predict.return_value = np.array([4.5, 4.6, 4.7])
        assert client.post("/predict/batch", json=self._batch_payload(3)).status_code == 200

    def test_n_samples_matches_input(self, client: TestClient) -> None:
        n = 5
        app_state.model.predict.return_value = np.array([4.5] * n)
        body = client.post("/predict/batch", json=self._batch_payload(n)).json()
        assert body["n_samples"] == n

    def test_request_ids_length_matches_predictions(self, client: TestClient) -> None:
        n = 4
        app_state.model.predict.return_value = np.array([4.5] * n)
        body = client.post("/predict/batch", json=self._batch_payload(n)).json()
        assert len(body["request_ids"]) == n

    def test_batch_predictions_all_logged(self, client: TestClient) -> None:
        n = 3
        app_state.model.predict.return_value = np.array([4.5] * n)
        client.post("/predict/batch", json=self._batch_payload(n))
        assert app_state.prediction_logger is not None
        log_path = app_state.prediction_logger.sink
        records = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == n

    def test_empty_samples_returns_422(self, client: TestClient) -> None:
        assert client.post("/predict/batch", json={"samples": []}).status_code == 422


# ─── Monitoring logs endpoint ─────────────────────────────────────────────────


class TestMonitoringLogsEndpoint:
    def test_returns_200_when_no_log_file(self, client: TestClient) -> None:
        """Before any predictions, the log file may not exist yet."""
        resp = client.get("/monitoring/logs")
        assert resp.status_code == 200

    def test_returns_count_after_predictions(self, client: TestClient) -> None:
        client.post("/predict", json=_sample_payload())
        client.post("/predict", json=_sample_payload())
        body = client.get("/monitoring/logs").json()
        assert body["n_predictions_logged"] == 2

    def test_prediction_stats_present_after_logging(self, client: TestClient) -> None:
        client.post("/predict", json=_sample_payload())
        body = client.get("/monitoring/logs").json()
        if body["n_predictions_logged"] > 0:
            stats = body["prediction_stats"]
            assert "mean" in stats
            assert "min" in stats
            assert "max" in stats


# ─── Degraded state ───────────────────────────────────────────────────────────


class TestDegradedState:
    def test_predict_503_when_no_model(self, client: TestClient) -> None:
        original = app_state.model
        app_state.model = None
        try:
            resp = client.post("/predict", json=_sample_payload())
            assert resp.status_code == 503
        finally:
            app_state.model = original

    def test_health_degraded_when_no_model(self, client: TestClient) -> None:
        original = app_state.model
        app_state.model = None
        try:
            body = client.get("/health").json()
            assert body["status"] == "degraded"
            assert body["model_loaded"] is False
        finally:
            app_state.model = original
