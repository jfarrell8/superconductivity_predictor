"""
src/features/engineer.py
─────────────────────────
Transforms raw superconductivity data into model-ready features.

Pipeline (matches EDA notebook decisions):
  1. BinningTransformer  — merge sparse bins in quasi-categorical columns
  2. TargetTransformer   — Box-Cox transform on critical_temp
  3. EncoderTransformer  — one-hot encode range_Valence (if applicable)
  4. FeaturePruner       — correlation + mutual-information filtering
  5. FeatureScaler       — StandardScaler (skips protected ordinal columns)

Each transformer follows the sklearn fit/transform protocol so this pipeline
is composable, testable, and easy to swap in/out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


# ─── Binning ─────────────────────────────────────────────────────────────────


@dataclass
class BinningConfig:
    """Configuration for a single quasi-categorical column binning operation."""

    column: str
    output_column: str
    rules: list[tuple[str, int | float, int | float]]
    """
    List of (operator, threshold, target_value) rules applied in order.
    Supported operators: "lte" (<=), "gte" (>=), "eq" (==).
    Rows not matched by any rule retain their original value.
    """


class BinningTransformer:
    """
    Collapses sparse bins in near-categorical integer columns.

    Directly encodes the EDA decisions:
      • number_of_elements: 1 → 2,   7/8/9 → 6
      • range_Valence:      0 → 1,   5/6   → 4
    """

    def __init__(self, configs: list[BinningConfig]) -> None:
        self.configs = configs

    def fit(self, df: pd.DataFrame) -> "BinningTransformer":  # stateless
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for cfg in self.configs:
            if cfg.column not in df.columns:
                logger.warning(f"BinningTransformer: column '{cfg.column}' not found, skipping.")
                continue
            series = df[cfg.column].copy()
            for op, threshold, target in cfg.rules:
                if op == "lte":
                    series = series.where(series > threshold, target)
                elif op == "gte":
                    series = series.where(series < threshold, target)
                elif op == "eq":
                    series = series.replace(threshold, target)
                else:
                    raise ValueError(f"Unknown operator '{op}' in BinningConfig.")
            df[cfg.output_column] = series
            logger.debug(f"Binned '{cfg.column}' → '{cfg.output_column}'")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ─── Target Transformation ────────────────────────────────────────────────────


class TargetTransformer:
    """
    Applies Box-Cox transformation to the regression target.

    Stores the fitted lambda so inverse-transform is available at inference
    time (required to convert model predictions back to raw Kelvin units).
    """

    def __init__(self, source_column: str, output_column: str) -> None:
        self.source_column = source_column
        self.output_column = output_column
        self._lambda: Optional[float] = None

    @property
    def fitted_lambda(self) -> float:
        if self._lambda is None:
            raise RuntimeError("TargetTransformer has not been fitted yet.")
        return self._lambda

    def fit(self, df: pd.DataFrame) -> "TargetTransformer":
        if (df[self.source_column] <= 0).any():
            raise ValueError(
                f"Box-Cox requires strictly positive values. "
                f"Column '{self.source_column}' contains zero or negative values."
            )
        _, lam = stats.boxcox(df[self.source_column])
        self._lambda = float(lam)
        logger.info(f"Box-Cox fitted lambda = {self._lambda:.4f}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._lambda is None:
            raise RuntimeError("Call fit() before transform().")
        df = df.copy()
        df[self.output_column] = stats.boxcox(df[self.source_column], lmbda=self._lambda)
        df.drop(columns=[self.source_column], inplace=True)
        logger.debug(f"Applied Box-Cox transform: '{self.source_column}' → '{self.output_column}'")
        return df

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Convert Box-Cox predictions back to original Kelvin scale."""
        lam = self.fitted_lambda
        if abs(lam) < 1e-10:
            return np.expm1(values)
        return np.power(np.maximum(lam * values + 1, 0), 1.0 / lam)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ─── Feature Pruner ───────────────────────────────────────────────────────────


@dataclass
class PrunerConfig:
    correlation_threshold: float = 0.8
    protected_columns: list[str] = field(default_factory=list)
    """Columns to exclude from correlation analysis (e.g. already-encoded categoricals)."""
    drop_columns: list[str] = field(default_factory=list)
    """Columns to drop unconditionally (e.g. low-MI dummy variables identified in EDA)."""


class FeaturePruner:
    """
    Reduces feature dimensionality via two strategies:

    1. Pairwise correlation pruning: For each pair with |r| > threshold,
       drop the column that appears later in the upper-triangle matrix
       (i.e. keep the first of the correlated pair).
    2. Explicit column dropping: Remove columns identified in EDA as
       uninformative (e.g. range_Valence dummies).

    Attributes
    ----------
    dropped_correlation : list[str]
        Columns removed by correlation pruning (populated after fit).
    """

    def __init__(self, config: PrunerConfig) -> None:
        self.config = config
        self.dropped_correlation: list[str] = []
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame, target_column: str) -> "FeaturePruner":
        analysis_df = df.drop(
            columns=self.config.protected_columns + [target_column],
            errors="ignore",
        )
        corr_matrix = analysis_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self.dropped_correlation = [
            col
            for col in upper.columns
            if any(upper[col] > self.config.correlation_threshold)
        ]
        logger.info(
            f"FeaturePruner: {len(self.dropped_correlation)} columns flagged by "
            f"correlation (threshold={self.config.correlation_threshold}): "
            f"{self.dropped_correlation}"
        )
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        to_drop = list(
            set(self.dropped_correlation + self.config.drop_columns) & set(df.columns)
        )
        df = df.drop(columns=to_drop)
        logger.info(f"FeaturePruner: dropped {len(to_drop)} columns total.")
        return df

    def fit_transform(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        return self.fit(df, target_column).transform(df)

    def mutual_info_ranking(
        self, df: pd.DataFrame, target_column: str
    ) -> pd.Series:
        """
        Compute mutual information between every feature and the target.
        Useful for post-pruning diagnostics and final top-k selection.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        mi = mutual_info_regression(X, y, random_state=42)
        return pd.Series(mi, index=X.columns).sort_values(ascending=False)


# ─── Scaler ───────────────────────────────────────────────────────────────────


class FeatureScaler:
    """
    Wraps sklearn's StandardScaler with support for skipping selected columns.

    The num_elements_simplified column is ordinal and physically meaningful
    as-is; scaling it would obscure that monotonic relationship with Tc.
    """

    def __init__(self, skip_columns: list[str] | None = None) -> None:
        self.skip_columns: list[str] = skip_columns or []
        self._scaler = StandardScaler()
        self._scale_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        self._scale_columns = [c for c in df.columns if c not in self.skip_columns]
        self._scaler.fit(df[self._scale_columns])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        scaled = self._scaler.transform(df[self._scale_columns])
        df[self._scale_columns] = scaled
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def scaler(self) -> StandardScaler:
        return self._scaler


# ─── High-level FeatureEngineer facade ───────────────────────────────────────


class FeatureEngineer:
    """
    Orchestrates the full feature engineering pipeline.

    Composes BinningTransformer → TargetTransformer → FeaturePruner →
    FeatureScaler into a single, serialisable object.

    Usage
    ─────
    >>> fe = FeatureEngineer.from_config(cfg)
    >>> X_train, y_train = fe.fit_transform(train_df)
    >>> X_test, _        = fe.transform(test_df)           # reuses fitted state
    """

    def __init__(
        self,
        binner: BinningTransformer,
        target_transformer: TargetTransformer,
        pruner: FeaturePruner,
        scaler: FeatureScaler,
        target_column: str,
        drop_raw_columns: list[str] | None = None,
    ) -> None:
        self.binner = binner
        self.target_transformer = target_transformer
        self.pruner = pruner
        self.scaler = scaler
        self.target_column = target_column
        self.drop_raw_columns = drop_raw_columns or []
        self._fitted = False

    @classmethod
    def from_config(cls, cfg: dict) -> "FeatureEngineer":
        """Construct a FeatureEngineer from a parsed YAML config dict."""
        fe_cfg = cfg["feature_engineering"]

        elem_cfg = fe_cfg["element_bins"]
        valence_cfg = fe_cfg["valence_bins"]

        binner = BinningTransformer(
            configs=[
                BinningConfig(
                    column=elem_cfg["column"],
                    output_column="num_elements_simplified",
                    rules=[
                        ("lte", elem_cfg["low_threshold"], elem_cfg["low_value"]),
                        ("gte", elem_cfg["high_threshold"], elem_cfg["high_value"]),
                    ],
                ),
                BinningConfig(
                    column=valence_cfg["column"],
                    output_column="rangeValence_simplified",
                    rules=[
                        ("eq", 0, valence_cfg["merge_zero_to"]),
                        ("gte", valence_cfg["high_threshold"], valence_cfg["high_value"]),
                    ],
                ),
            ]
        )

        target_transformer = TargetTransformer(
            source_column=fe_cfg["target_column"],
            output_column=fe_cfg["engineered_target_column"],
        )

        pruner = FeaturePruner(
            config=PrunerConfig(
                correlation_threshold=fe_cfg["correlation_threshold"],
                protected_columns=fe_cfg["protected_columns"],
                drop_columns=fe_cfg["drop_low_mi_columns"],
            )
        )

        scaler = FeatureScaler(
            skip_columns=cfg["preprocessing"]["columns_to_skip_scaling"]
        )

        return cls(
            binner=binner,
            target_transformer=target_transformer,
            pruner=pruner,
            scaler=scaler,
            target_column=fe_cfg["engineered_target_column"],
            drop_raw_columns=[
                fe_cfg["element_bins"]["column"],
                fe_cfg["valence_bins"]["column"],
            ],
        )

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fit all transformers on df and return (X_scaled, y)."""
        logger.info("FeatureEngineer: starting fit_transform")

        # 1. Bin quasi-categorical columns
        df = self.binner.fit_transform(df)

        # 2. One-hot encode range_Valence (fitted during EDA; dropped later)
        if "rangeValence_simplified" in df.columns:
            dummies = pd.get_dummies(
                df["rangeValence_simplified"], prefix="rangeValence"
            )
            df = pd.concat(
                [df.drop(columns=["rangeValence_simplified"]), dummies], axis=1
            )

        # 3. Drop the original raw columns
        df.drop(columns=[c for c in self.drop_raw_columns if c in df.columns], inplace=True)

        # 4. Box-Cox transform target
        df = self.target_transformer.fit_transform(df)

        # 5. Correlation + MI pruning
        df = self.pruner.fit_transform(df, target_column=self.target_column)

        # 6. Separate X and y before scaling
        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])

        # 7. Scale features
        X_scaled = self.scaler.fit_transform(X)

        self._fitted = True
        logger.info(
            f"FeatureEngineer: produced {X_scaled.shape[1]} features for "
            f"{len(X_scaled):,} samples."
        )
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Apply fitted transformations to new data (inference path)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")

        df = self.binner.transform(df)

        if "rangeValence_simplified" in df.columns:
            dummies = pd.get_dummies(
                df["rangeValence_simplified"], prefix="rangeValence"
            )
            df = pd.concat(
                [df.drop(columns=["rangeValence_simplified"]), dummies], axis=1
            )

        df.drop(columns=[c for c in self.drop_raw_columns if c in df.columns], inplace=True)

        y: Optional[pd.Series] = None
        if self.target_column.replace("_boxcox", "") in df.columns:
            # raw target present — transform it
            df = self.target_transformer.transform(df)
            y = df[self.target_column]
            df = df.drop(columns=[self.target_column])
        elif self.target_column in df.columns:
            y = df[self.target_column]
            df = df.drop(columns=[self.target_column])

        df = self.pruner.transform(df)
        X_scaled = self.scaler.transform(df)
        return X_scaled, y
