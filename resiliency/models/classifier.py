"""
XGBoost default risk classifier.

Wraps an XGBClassifier in a production-grade class that handles
preprocessing, calibration, threshold tuning, serialisation, and
interpretability.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from resiliency.data.generator import FEATURE_COLS, LABEL_COL
from resiliency.utils.preprocessing import FeaturePreprocessor


# ---------------------------------------------------------------------------
# Platt-scaling wrapper (replaces CalibratedClassifierCV(cv="prefit"),
# which was removed in sklearn 1.4)
# ---------------------------------------------------------------------------

class _PreFitCalibrated:
    """Wraps a pre-fitted estimator with a sigmoid (Platt) calibrator."""

    def __init__(self, estimator, platt: LogisticRegression) -> None:
        self._estimator = estimator
        self._platt = platt

    def predict_proba(self, X) -> np.ndarray:
        raw = self._estimator.predict_proba(X)[:, 1].reshape(-1, 1)
        pos = self._platt.predict_proba(raw)[:, 1]
        return np.column_stack([1 - pos, pos])


# ---------------------------------------------------------------------------
# Default hyper-parameters (tuned for class-imbalanced credit data)
# ---------------------------------------------------------------------------
DEFAULT_XGB_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 3.5,   # compensates for ~22% default rate
    # "use_label_encoder": False,
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


class DefaultRiskClassifier:
    """
    XGBoost binary classifier that predicts 12-month default probability.

    Parameters
    ----------
    xgb_params : dict, optional
        XGBoost hyper-parameters.  Defaults to ``DEFAULT_XGB_PARAMS``.
    calibrate : bool
        If True, apply Platt scaling (sigmoid calibration) after training
        to produce well-calibrated probabilities.
    classification_threshold : float
        Decision threshold for binary prediction. Defaults to 0.5.

    Examples
    --------
    >>> clf = DefaultRiskClassifier()
    >>> clf.fit(X_train, y_train)
    >>> probs = clf.predict_proba(X_test)
    >>> preds = clf.predict(X_test)
    """

    def __init__(
        self,
        xgb_params: Optional[dict] = None,
        calibrate: bool = True,
        classification_threshold: float = 0.50,
    ) -> None:
        self.xgb_params = xgb_params or DEFAULT_XGB_PARAMS
        self.calibrate = calibrate
        self.classification_threshold = classification_threshold

        self._preprocessor = FeaturePreprocessor(scale_numeric=False, add_interactions=True)
        self._model: Optional[XGBClassifier | _PreFitCalibrated] = None
        self._raw_xgb: Optional[XGBClassifier] = None
        self._feature_names: list[str] = []
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        eval_set: Optional[tuple] = None,
        verbose: bool = False,
    ) -> "DefaultRiskClassifier":
        """
        Fit the preprocessor and XGBoost classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame with columns matching ``FEATURE_COLS``.
        y : array-like
            Binary default labels.
        eval_set : tuple, optional
            (X_val, y_val) for early stopping.
        verbose : bool
            Whether XGBoost should print training logs.

        Returns
        -------
        DefaultRiskClassifier
            self
        """
        logger.info("Fitting DefaultRiskClassifier on {} samples", len(X))

        # Preprocess
        self._preprocessor.fit(X, y)
        X_proc = self._preprocessor.transform(X)
        self._feature_names = self._preprocessor.get_feature_names_out()

        # Build base XGBoost
        xgb = XGBClassifier(**self.xgb_params)
        self._raw_xgb = xgb

        fit_kwargs: dict = {}
        if eval_set is not None:
            X_val_proc = self._preprocessor.transform(eval_set[0])
            fit_kwargs["eval_set"] = [(X_val_proc, eval_set[1])]
            fit_kwargs["verbose"] = verbose

        if self.calibrate:
            # Split off a calibration holdout (cv="prefit" removed in sklearn 1.4)
            X_fit, X_cal, y_fit, y_cal = train_test_split(
                X_proc, y, test_size=0.2, stratify=y, random_state=42
            )
            xgb.fit(X_fit, y_fit, **fit_kwargs)
            logger.info("Calibrating probabilities with sigmoid calibration")
            raw_cal = xgb.predict_proba(X_cal)[:, 1].reshape(-1, 1)
            platt = LogisticRegression(C=1e10)
            platt.fit(raw_cal, y_cal)
            self._model = _PreFitCalibrated(xgb, platt)
        else:
            xgb.fit(X_proc, y, **fit_kwargs)
            self._model = xgb

        self.is_fitted = True
        # train_auc = self._cv_score(X_proc, y)
        # logger.success("Training complete — CV AUC={:.4f}", train_auc)
        return self

    def _cv_score(self, X: np.ndarray, y, cv: int = 5) -> float:
        xgb_tmp = XGBClassifier(**self.xgb_params)
        scores = cross_val_score(
            xgb_tmp, X, y,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring="roc_auc",
            n_jobs=-1,
        )
        return float(scores.mean())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return predicted default probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Probability of default (positive class).
        """
        self._check_fitted()
        X_proc = self._preprocessor.transform(X)
        return self._model.predict_proba(X_proc)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return binary default predictions using ``classification_threshold``.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba >= self.classification_threshold).astype(int)

    def predict_with_score(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with probability, prediction, and risk tier.

        Risk tiers: LOW (<30%), MEDIUM (30-60%), HIGH (>60%)
        """
        proba = self.predict_proba(X)
        pred = (proba >= self.classification_threshold).astype(int)
        tier = np.where(proba < 0.30, "LOW", np.where(proba < 0.60, "MEDIUM", "HIGH"))
        return pd.DataFrame(
            {"default_probability": proba, "default_prediction": pred, "risk_tier": tier}
        )

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self) -> np.ndarray:
        """XGBoost gain-based feature importances."""
        self._check_fitted()
        if self.calibrate:
            return self._raw_xgb.feature_importances_
        return self._model.feature_importances_

    @property
    def feature_names_(self) -> list[str]:
        return self._feature_names

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the model to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Model saved → {}", path)

    @classmethod
    def load(cls, path: str | Path) -> "DefaultRiskClassifier":
        """Load a persisted model from disk."""
        obj = joblib.load(path)
        logger.info("Model loaded ← {}", path)
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return (
            f"DefaultRiskClassifier("
            f"calibrate={self.calibrate}, "
            f"threshold={self.classification_threshold}, "
            f"status={status})"
        )
