"""Unit tests for the Q-learning RL agent."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resiliency.data.generator import CustomerDataGenerator, GeneratorConfig
from resiliency.models.rl_agent import (
    QLearningAgent,
    OfferType,
    discretise_state,
    extract_rl_state,
    N_ACTIONS,
)


@pytest.fixture(scope="module")
def small_df():
    gen = CustomerDataGenerator(GeneratorConfig(n_samples=300, random_seed=1))
    return gen.generate()


@pytest.fixture(scope="module")
def trained_agent(small_df):
    agent = QLearningAgent(epsilon_decay=0.99)
    agent.train(small_df, n_episodes=500, log_every=200)
    return agent


class TestOfferTypes:
    def test_all_offers_defined(self):
        assert len(OfferType) == N_ACTIONS

    def test_offer_values(self):
        assert OfferType.NO_ACTION == 0
        assert OfferType.SETTLEMENT_OFFER == 3


class TestStateExtraction:
    def test_extract_rl_state_shape(self, small_df):
        row = small_df.iloc[0]
        state_vec = extract_rl_state(row)
        assert state_vec.shape == (10,)  # len(RL_STATE_FEATURES)

    def test_extract_rl_state_range(self, small_df):
        for i in range(min(50, len(small_df))):
            vec = extract_rl_state(small_df.iloc[i])
            assert np.all(vec >= 0.0) and np.all(vec <= 1.0)

    def test_discretise_state_returns_tuple(self, small_df):
        row = small_df.iloc[0]
        state = discretise_state(row)
        assert isinstance(state, tuple)
        assert len(state) == 10


class TestQLearningAgent:
    def test_trains_without_error(self, trained_agent):
        assert trained_agent.is_trained

    def test_q_table_populated(self, trained_agent):
        assert len(trained_agent._q) > 0

    def test_reward_history_length(self, trained_agent):
        assert len(trained_agent.reward_history) == 500

    def test_act_greedy_returns_valid_action(self, trained_agent, small_df):
        state = discretise_state(small_df.iloc[0])
        action = trained_agent.act(state, greedy=True)
        assert 0 <= action < N_ACTIONS

    def test_recommend_returns_expected_keys(self, trained_agent, small_df):
        customer = small_df.iloc[0].to_dict()
        rec = trained_agent.recommend(customer, default_prob=0.6)
        required_keys = {"action", "offer_type", "offer_label", "q_values", "confidence", "default_probability"}
        assert required_keys <= set(rec.keys())

    def test_recommend_confidence_in_unit_interval(self, trained_agent, small_df):
        for i in range(10):
            customer = small_df.iloc[i].to_dict()
            rec = trained_agent.recommend(customer, default_prob=0.5)
            assert 0.0 <= rec["confidence"] <= 1.0

    def test_recommend_action_matches_q_values(self, trained_agent, small_df):
        customer = small_df.iloc[0].to_dict()
        rec = trained_agent.recommend(customer, default_prob=0.5)
        best_offer = max(rec["q_values"], key=rec["q_values"].get)
        assert best_offer == rec["offer_type"]

    def test_untrained_raises(self):
        agent = QLearningAgent()
        with pytest.raises(RuntimeError, match="not trained"):
            agent.recommend({}, 0.5)

    def test_save_load_roundtrip(self, trained_agent, small_df, tmp_path):
        path = tmp_path / "agent.pkl"
        trained_agent.save(path)
        loaded = QLearningAgent.load(path)
        customer = small_df.iloc[0].to_dict()
        orig = trained_agent.recommend(customer, 0.5)
        loaded_rec = loaded.recommend(customer, 0.5)
        assert orig["action"] == loaded_rec["action"]
