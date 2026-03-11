"""
Feature preprocessing and pipeline construction.

Provides a scikit-learn compatible transformer that encodes,
scales, and engineers features for the default risk classifier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Optional

from resiliency.data.generator import FEATURE_COLS, LABEL_COL, HARDSHIP_SEVERITY_COL


# Features that should not be scaled (already binary / categorical)
BINARY_COLS = ["has_bankruptcy", "requested_hardship_program", "is_in_deferment"]
CATEGORICAL_COLS = ["employment_status", "state_code"]
NUMERIC_COLS = [
    c for c in FEATURE_COLS if c not in BINARY_COLS + CATEGORICAL_COLS
]


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer for customer hardship features.

    Steps applied
    -------------
    1. Clip extreme values at 1st / 99th percentile.
    2. Add derived interaction features.
    3. Standard-scale continuous numeric columns.
    4. Leave binary and encoded-categorical columns unchanged.

    Parameters
    ----------
    scale_numeric : bool
        Whether to apply StandardScaler to numeric columns.
    add_interactions : bool
        Whether to add interaction terms.
    """

    def __init__(
        self,
        scale_numeric: bool = True,
        add_interactions: bool = True,
    ) -> None:
        self.scale_numeric = scale_numeric
        self.add_interactions = add_interactions
        self._scaler = StandardScaler()
        self._clip_bounds: dict[str, tuple[float, float]] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeaturePreprocessor":
        """Compute clip bounds and scaler statistics from training data."""
        df = self._select_features(X)
        # Compute clip bounds per numeric column
        for col in NUMERIC_COLS:
            if col in df.columns:
                lo = float(np.percentile(df[col].dropna(), 1))
                hi = float(np.percentile(df[col].dropna(), 99))
                self._clip_bounds[col] = (lo, hi)
        # Fit scaler on clipped numerics
        if self.scale_numeric:
            clipped = self._clip(df)
            self._scaler.fit(clipped[NUMERIC_COLS])
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply all preprocessing steps, return numpy array."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        df = self._select_features(X).copy()
        df = self._clip(df)
        if self.add_interactions:
            df = self._engineer_interactions(df)
        if self.scale_numeric:
            numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
            df[numeric_present] = self._scaler.transform(df[numeric_present])
        return df.values.astype(np.float32)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return output feature names."""
        cols = list(FEATURE_COLS)
        if self.add_interactions:
            cols += self._interaction_names()
        return cols

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_features(X: pd.DataFrame) -> pd.DataFrame:
        """Keep only recognised feature columns, fill missing with 0."""
        present = [c for c in FEATURE_COLS if c in X.columns]
        out = X[present].copy()
        for c in FEATURE_COLS:
            if c not in out.columns:
                out[c] = 0.0
        return out[FEATURE_COLS]

    def _clip(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (lo, hi) in self._clip_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        return df

    @staticmethod
    def _engineer_interactions(df: pd.DataFrame) -> pd.DataFrame:
        """Add domain-driven interaction features."""
        df = df.copy()
        # Stressed balance: high utilization + many missed payments
        df["util_x_missed"] = (
            df["credit_utilization_pct"] * df["num_missed_payments_12m"]
        )
        # Income stress: debt load relative to income
        df["income_stress"] = df["debt_to_income_ratio"] * df["months_delinquent"]
        # Recency of hardship
        df["recent_hardship"] = (
            df["months_since_last_payment"] * df["consecutive_missed_payments"]
        )
        return df

    @staticmethod
    def _interaction_names() -> list[str]:
        return ["util_x_missed", "income_stress", "recent_hardship"]


def build_pipeline(classifier) -> Pipeline:
    """
    Wrap a classifier in a preprocessing pipeline.

    Parameters
    ----------
    classifier
        Any scikit-learn compatible estimator.

    Returns
    -------
    Pipeline
        sklearn Pipeline with FeaturePreprocessor + classifier.
    """
    return Pipeline(
        steps=[
            ("preprocessor", FeaturePreprocessor()),
            ("classifier", classifier),
        ]
    )
