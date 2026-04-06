# tests/conftest.py
"""
Shared pytest fixtures available to all test modules.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_train_df() -> pd.DataFrame:
    """
    Session-scoped synthetic DataFrame that mirrors the superconductivity
    dataset schema. Reused across test modules to avoid re-generating.
    """
    rng = np.random.default_rng(99)
    n = 500

    # Simulate 81-column superconductivity-like data
    feature_cols = {f"feat_{i}": rng.uniform(0, 100, size=n) for i in range(79)}
    return pd.DataFrame(
        {
            "critical_temp": rng.uniform(1.0, 140.0, size=n),  # strictly positive
            "number_of_elements": rng.integers(1, 10, size=n),
            "range_Valence": rng.integers(0, 7, size=n),
            **feature_cols,
        }
    )


@pytest.fixture()
def small_feature_df() -> pd.DataFrame:
    """Small DataFrame of already-engineered features for model/eval tests."""
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame(
        {f"feature_{i}": rng.normal(size=n) for i in range(10)} | {"target": rng.normal(size=n)}
    )


@pytest.fixture(autouse=True, scope="session")
def disable_prefect_server() -> None:
    """Prevent Prefect from trying to connect to a server during tests."""
    original = os.environ.get("PREFECT_API_URL")
    os.environ["PREFECT_API_URL"] = ""
    yield
    if original is None:
        os.environ.pop("PREFECT_API_URL", None)
    else:
        os.environ["PREFECT_API_URL"] = original


@pytest.fixture(autouse=True, scope="session")
def suppress_prefect_shutdown_noise() -> None:
    """Suppress Prefect's noisy shutdown logging after tests complete."""
    yield
    # Silence Prefect's internal loggers during teardown
    logging.getLogger("prefect.server").setLevel(logging.CRITICAL)
    logging.getLogger("prefect").setLevel(logging.CRITICAL)
