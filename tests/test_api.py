"""Integration tests for the FastAPI scoring endpoints."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from resiliency.data.generator import CustomerDataGenerator, GeneratorConfig, LABEL_COL
from resiliency.models.classifier import DefaultRiskClassifier
from resiliency.models.rl_agent import QLearningAgent


# ---------------------------------------------------------------------------
# Sample customer payload
# ---------------------------------------------------------------------------

SAMPLE_CUSTOMER = {
    "age": 38,
    "annual_income": 42000,
    "employment_status": 1,
    "household_size": 3,
    "state_code": 12,
    "credit_score": 580,
    "num_open_accounts": 4,
    "credit_utilization_pct": 0.82,
    "credit_limit": 8000,
    "current_balance": 6560,
    "months_since_opened": 48,
    "months_delinquent": 3,
    "num_missed_payments_12m": 4,
    "consecutive_missed_payments": 2,
    "min_payment_ratio": 0.6,
    "months_since_last_payment": 3,
    "debt_to_income_ratio": 0.95,
    "total_debt": 39900,
    "num_collections": 1,
    "has_bankruptcy": 0,
    "requested_hardship_program": 1,
    "is_in_deferment": 0,
    "num_prior_hardship_programs": 1,
    "hardship_severity": 1,
}


@pytest.fixture(scope="module")
def trained_models():
    """Train minimal models for API testing."""
    gen = CustomerDataGenerator(GeneratorConfig(n_samples=800, random_seed=99))
    df = gen.generate()
    train, test = gen.train_test_split(df, test_size=0.2)

    X_train = train.drop(columns=[LABEL_COL, "hardship_severity"])
    y_train = train[LABEL_COL]

    clf = DefaultRiskClassifier(calibrate=False)  # faster for tests
    clf.fit(X_train, y_train)

    all_X = df.drop(columns=[LABEL_COL, "hardship_severity"])
    all_probs = clf.predict_proba(all_X)

    agent = QLearningAgent(epsilon_decay=0.99)
    agent.train(df, default_probs=all_probs, n_episodes=500)

    return clf, agent


@pytest.fixture(scope="module")
def client(trained_models):
    """FastAPI test client with models injected into registry."""
    from api.main import app, _registry

    clf, agent = trained_models
    _registry.classifier = clf
    _registry.rl_agent = agent

    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["classifier_loaded"] is True
        assert body["rl_agent_loaded"] is True


class TestScoreEndpoint:
    def test_score_returns_200(self, client):
        resp = client.post("/score", json=SAMPLE_CUSTOMER)
        assert resp.status_code == 200

    def test_score_response_schema(self, client):
        resp = client.post("/score", json=SAMPLE_CUSTOMER)
        body = resp.json()
        assert "default_probability" in body
        assert "default_prediction" in body
        assert "risk_tier" in body

    def test_probability_in_unit_interval(self, client):
        resp = client.post("/score", json=SAMPLE_CUSTOMER)
        prob = resp.json()["default_probability"]
        assert 0.0 <= prob <= 1.0

    def test_prediction_binary(self, client):
        resp = client.post("/score", json=SAMPLE_CUSTOMER)
        assert resp.json()["default_prediction"] in (0, 1)

    def test_risk_tier_valid(self, client):
        resp = client.post("/score", json=SAMPLE_CUSTOMER)
        assert resp.json()["risk_tier"] in ("LOW", "MEDIUM", "HIGH")

    def test_invalid_credit_score_rejected(self, client):
        bad = {**SAMPLE_CUSTOMER, "credit_score": 9999}
        resp = client.post("/score", json=bad)
        assert resp.status_code == 422


class TestBatchScoreEndpoint:
    def test_batch_two_customers(self, client):
        payload = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
        resp = client.post("/score/batch", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert len(body["results"]) == 2

    def test_empty_batch_rejected(self, client):
        resp = client.post("/score/batch", json={"customers": []})
        assert resp.status_code == 422


class TestRecommendEndpoint:
    def test_recommend_returns_200(self, client):
        resp = client.post("/recommend", json=SAMPLE_CUSTOMER)
        assert resp.status_code == 200

    def test_recommend_schema(self, client):
        resp = client.post("/recommend", json=SAMPLE_CUSTOMER)
        body = resp.json()
        for key in ("action", "offer_type", "offer_label", "confidence", "q_values"):
            assert key in body

    def test_confidence_in_unit_interval(self, client):
        resp = client.post("/recommend", json=SAMPLE_CUSTOMER)
        conf = resp.json()["confidence"]
        assert 0.0 <= conf <= 1.0


class TestCombinedEndpoint:
    def test_combined_returns_both(self, client):
        resp = client.post("/score-and-recommend", json=SAMPLE_CUSTOMER)
        assert resp.status_code == 200
        body = resp.json()
        assert "score" in body
        assert "recommendation" in body
        assert "default_probability" in body["score"]
        assert "offer_label" in body["recommendation"]
