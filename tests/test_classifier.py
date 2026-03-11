"""Unit tests for the XGBoost default risk classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resiliency.data.generator import CustomerDataGenerator, GeneratorConfig, LABEL_COL
from resiliency.models.classifier import DefaultRiskClassifier


@pytest.fixture(scope="module")
def dataset():
    gen = CustomerDataGenerator(GeneratorConfig(n_samples=1_000, random_seed=42))
    df = gen.generate()
    train, test = gen.train_test_split(df, test_size=0.2)
    X_train = train.drop(columns=[LABEL_COL, "hardship_severity"])
    y_train = train[LABEL_COL]
    X_test = test.drop(columns=[LABEL_COL, "hardship_severity"])
    y_test = test[LABEL_COL]
    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
def fitted_clf(dataset):
    X_train, y_train, X_test, y_test = dataset
    clf = DefaultRiskClassifier(calibrate=True, classification_threshold=0.5)
    clf.fit(X_train, y_train)
    return clf


class TestClassifierFitting:
    def test_fits_without_error(self, fitted_clf):
        assert fitted_clf.is_fitted

    def test_predict_proba_shape(self, fitted_clf, dataset):
        _, _, X_test, _ = dataset
        proba = fitted_clf.predict_proba(X_test)
        assert proba.shape == (len(X_test),)

    def test_probabilities_in_unit_interval(self, fitted_clf, dataset):
        _, _, X_test, _ = dataset
        proba = fitted_clf.predict_proba(X_test)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_predict_binary(self, fitted_clf, dataset):
        _, _, X_test, _ = dataset
        preds = fitted_clf.predict(X_test)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_roc_auc_above_threshold(self, fitted_clf, dataset):
        from sklearn.metrics import roc_auc_score
        _, _, X_test, y_test = dataset
        proba = fitted_clf.predict_proba(X_test)
        auc = roc_auc_score(y_test, proba)
        assert auc >= 0.70, f"ROC-AUC {auc:.3f} below minimum threshold 0.70"

    def test_feature_importances_length(self, fitted_clf):
        fi = fitted_clf.feature_importances_
        fn = fitted_clf.feature_names_
        assert len(fi) == len(fn)

    def test_feature_importances_non_negative(self, fitted_clf):
        assert np.all(fitted_clf.feature_importances_ >= 0)


class TestPredictWithScore:
    def test_columns(self, fitted_clf, dataset):
        _, _, X_test, _ = dataset
        result = fitted_clf.predict_with_score(X_test)
        assert {"default_probability", "default_prediction", "risk_tier"} <= set(result.columns)

    def test_risk_tiers_valid(self, fitted_clf, dataset):
        _, _, X_test, _ = dataset
        result = fitted_clf.predict_with_score(X_test)
        assert result["risk_tier"].isin(["LOW", "MEDIUM", "HIGH"]).all()


class TestUnfitted:
    def test_predict_raises_before_fit(self):
        clf = DefaultRiskClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba(pd.DataFrame([{"credit_score": 600}]))


class TestSerialisation:
    def test_save_load_roundtrip(self, fitted_clf, dataset, tmp_path):
        _, _, X_test, _ = dataset
        path = tmp_path / "clf.pkl"
        fitted_clf.save(path)
        loaded = DefaultRiskClassifier.load(path)
        orig_proba = fitted_clf.predict_proba(X_test)
        loaded_proba = loaded.predict_proba(X_test)
        np.testing.assert_array_almost_equal(orig_proba, loaded_proba, decimal=5)
