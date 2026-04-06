"""
tests/unit/test_sagemaker_inference.py
────────────────────────────────────────
Unit tests for src/api/sagemaker/inference.py.

These tests mock joblib.load so they run without real model artifacts —
the inference script logic is tested in full isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ─── Fixtures ─────────────────────────────────────────────────────────────────


TOP_FEATURES = [
    "mean_atomic_mass",
    "wtd_mean_atomic_mass",
    "gmean_atomic_mass",
    "mean_fie",
    "mean_atomic_radius",
    "wtd_mean_atomic_radius",
    "mean_Density",
    "wtd_mean_Density",
    "mean_ElectronAffinity",
    "wtd_mean_ElectronAffinity",
    "mean_FusionHeat",
    "wtd_mean_FusionHeat",
    "mean_ThermalConductivity",
    "wtd_mean_ThermalConductivity",
    "num_elements_simplified",
]

SAMPLE_INPUT = {feat: float(i + 1.0) for i, feat in enumerate(TOP_FEATURES)}


# @pytest.fixture()
# def mock_model_dir(tmp_path: Path) -> Path:
#     """Create a fake model directory with required files."""
#     mock_model = MagicMock()
#     mock_model.predict.return_value = np.array([4.312])

#     import joblib

#     joblib.dump(mock_model, tmp_path / "best_model_top15.pkl")
#     (tmp_path / "top_features.json").write_text(json.dumps(TOP_FEATURES))
#     return tmp_path

@pytest.fixture()
def mock_model_dir(tmp_path: Path) -> Path:
    import joblib
    from sklearn.linear_model import LinearRegression

    # Use a real fitted model — MagicMock is not picklable
    model = LinearRegression()
    X = pd.DataFrame({feat: [1.0, 2.0, 3.0] for feat in TOP_FEATURES})
    y = pd.Series([4.1, 4.5, 4.9])
    model.fit(X, y)

    joblib.dump(model, tmp_path / "best_model_top15.pkl")
    (tmp_path / "top_features.json").write_text(json.dumps(TOP_FEATURES))
    return tmp_path


@pytest.fixture()
def model_artifacts(mock_model_dir: Path) -> dict:
    from src.api.sagemaker.inference import model_fn

    return model_fn(str(mock_model_dir))


# ─── model_fn ─────────────────────────────────────────────────────────────────


class TestModelFn:
    def test_returns_dict_with_model_and_features(self, mock_model_dir: Path) -> None:
        from src.api.sagemaker.inference import model_fn

        artifacts = model_fn(str(mock_model_dir))
        assert "model" in artifacts
        assert "top_features" in artifacts

    def test_top_features_matches_json(self, mock_model_dir: Path) -> None:
        from src.api.sagemaker.inference import model_fn

        artifacts = model_fn(str(mock_model_dir))
        assert artifacts["top_features"] == TOP_FEATURES

    def test_raises_if_model_missing(self, tmp_path: Path) -> None:
        (tmp_path / "top_features.json").write_text(json.dumps(TOP_FEATURES))
        with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
            from src.api.sagemaker.inference import model_fn

            model_fn(str(tmp_path))


# ─── input_fn ─────────────────────────────────────────────────────────────────


class TestInputFn:
    def test_json_single_sample(self) -> None:
        from src.api.sagemaker.inference import input_fn

        body = json.dumps({"features": SAMPLE_INPUT})
        df = input_fn(body, "application/json")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert set(TOP_FEATURES).issubset(set(df.columns))

    def test_json_instances_batch(self) -> None:
        from src.api.sagemaker.inference import input_fn

        body = json.dumps({"instances": [SAMPLE_INPUT, SAMPLE_INPUT]})
        df = input_fn(body, "application/json")
        assert len(df) == 2

    def test_json_bare_dict(self) -> None:
        from src.api.sagemaker.inference import input_fn

        body = json.dumps(SAMPLE_INPUT)
        df = input_fn(body, "application/json")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_csv_input(self) -> None:
        from src.api.sagemaker.inference import input_fn

        header = ",".join(TOP_FEATURES)
        values = ",".join(str(float(i + 1)) for i in range(len(TOP_FEATURES)))
        body = f"{header}\n{values}"
        df = input_fn(body, "text/csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_unsupported_content_type_raises(self) -> None:
        from src.api.sagemaker.inference import input_fn

        with pytest.raises(ValueError, match="Unsupported content type"):
            input_fn(b"data", "application/octet-stream")


# ─── predict_fn ───────────────────────────────────────────────────────────────


class TestPredictFn:
    def test_returns_ndarray(self, model_artifacts: dict) -> None:
        from src.api.sagemaker.inference import predict_fn

        df = pd.DataFrame([SAMPLE_INPUT])
        result = predict_fn(df, model_artifacts)
        assert isinstance(result, np.ndarray)

    def test_extra_columns_do_not_cause_error(
        self, model_artifacts: dict
    ) -> None:
        from src.api.sagemaker.inference import predict_fn

        extra = {**SAMPLE_INPUT, "extra_irrelevant_col": 999.0}
        df = pd.DataFrame([extra])
        result = predict_fn(df, model_artifacts)
        assert isinstance(result, np.ndarray)
        assert len(result) == 1

    def test_raises_on_missing_features(self, model_artifacts: dict) -> None:
        from src.api.sagemaker.inference import predict_fn

        incomplete = {k: v for k, v in SAMPLE_INPUT.items() if k != "mean_atomic_mass"}
        df = pd.DataFrame([incomplete])
        with pytest.raises(ValueError, match="missing required features"):
            predict_fn(df, model_artifacts)


# ─── output_fn ────────────────────────────────────────────────────────────────


class TestOutputFn:
    def test_json_output(self) -> None:
        from src.api.sagemaker.inference import output_fn

        preds = np.array([4.312, 3.891])
        body, content_type = output_fn(preds, "application/json")
        assert content_type == "application/json"
        parsed = json.loads(body)
        assert "predictions" in parsed
        assert len(parsed["predictions"]) == 2

    def test_csv_output(self) -> None:
        from src.api.sagemaker.inference import output_fn

        preds = np.array([4.312, 3.891])
        body, content_type = output_fn(preds, "text/csv")
        assert content_type == "text/csv"
        lines = body.strip().split("\n")
        assert len(lines) == 2

    def test_wildcard_accept_returns_json(self) -> None:
        from src.api.sagemaker.inference import output_fn

        body, content_type = output_fn(np.array([1.0]), "*/*")
        assert content_type == "application/json"

    def test_unsupported_accept_raises(self) -> None:
        from src.api.sagemaker.inference import output_fn

        with pytest.raises(ValueError, match="Unsupported accept type"):
            output_fn(np.array([1.0]), "application/xml")
