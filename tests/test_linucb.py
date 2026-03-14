"""Unit tests for the LinUCB contextual bandit agent."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from resiliency.models.linucb import (
    ARM_LABELS,
    N_ARMS,
    LinUCBAgent,
    LinUCBArm,
    _ArmState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def context(rng) -> np.ndarray:
    """A single normalised 10-dim context vector (float32)."""
    return rng.uniform(0.0, 1.0, size=10).astype(np.float32)


@pytest.fixture(scope="module")
def context_batch(rng) -> np.ndarray:
    """Batch of 50 normalised context vectors."""
    return rng.uniform(0.0, 1.0, size=(50, 10)).astype(np.float32)


@pytest.fixture(scope="module")
def trained_agent(rng, context_batch) -> LinUCBAgent:
    """Agent trained on 200 random (context, reward) pairs."""
    agent = LinUCBAgent(n_features=10, alpha=1.0)
    rewards = rng.normal(1.0, 0.3, size=len(context_batch))
    for i, ctx in enumerate(context_batch):
        arm = agent.select_action(ctx)
        agent.update(ctx, arm, float(rewards[i]))
    # Extra pass so every arm gets multiple updates
    for i, ctx in enumerate(context_batch):
        arm = agent.select_action(ctx)
        agent.update(ctx, arm, float(rewards[i]))
    return agent


# ---------------------------------------------------------------------------
# Arm enum & labels
# ---------------------------------------------------------------------------

class TestLinUCBArm:
    def test_four_arms_defined(self):
        assert len(LinUCBArm) == 4

    def test_arm_values(self):
        assert LinUCBArm.PAYMENT_PLAN == 0
        assert LinUCBArm.SETTLEMENT_30PCT == 1
        assert LinUCBArm.SETTLEMENT_50PCT == 2
        assert LinUCBArm.HARDSHIP_PROGRAM == 3

    def test_arm_labels_complete(self):
        for arm in LinUCBArm:
            assert arm in ARM_LABELS
            assert isinstance(ARM_LABELS[arm], str)
            assert len(ARM_LABELS[arm]) > 0


# ---------------------------------------------------------------------------
# _ArmState internals
# ---------------------------------------------------------------------------

class TestArmState:
    def test_initial_A_is_identity(self):
        arm = _ArmState(n_features=5)
        np.testing.assert_array_equal(arm.A, np.eye(5))

    def test_initial_b_is_zero(self):
        arm = _ArmState(n_features=5)
        np.testing.assert_array_equal(arm.b, np.zeros(5))

    def test_initial_n_updates_is_zero(self):
        assert _ArmState(n_features=5).n_updates == 0

    def test_theta_initial_is_zero(self):
        arm = _ArmState(n_features=4)
        np.testing.assert_array_almost_equal(arm.theta(), np.zeros(4))

    def test_update_increments_n_updates(self):
        arm = _ArmState(n_features=4)
        x = np.ones(4, dtype=np.float64)
        arm.update(x, reward=1.0)
        assert arm.n_updates == 1

    def test_update_modifies_A_and_b(self):
        arm = _ArmState(n_features=4)
        x = np.array([1.0, 0.0, 0.0, 0.0])
        arm.update(x, reward=2.0)
        # A should no longer be identity
        assert not np.allclose(arm.A, np.eye(4))
        # b should be non-zero in the first position
        assert arm.b[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestLinUCBInit:
    def test_default_params(self):
        agent = LinUCBAgent()
        assert agent.n_arms == N_ARMS
        assert agent.n_features == 10
        assert agent.alpha == 1.0

    def test_custom_params(self):
        agent = LinUCBAgent(n_arms=3, n_features=5, alpha=2.5)
        assert agent.n_arms == 3
        assert agent.n_features == 5
        assert agent.alpha == pytest.approx(2.5)

    def test_arms_list_length(self):
        agent = LinUCBAgent(n_arms=4, n_features=10)
        assert len(agent._arms) == 4

    def test_initial_A_matrices_are_identity(self):
        agent = LinUCBAgent(n_features=6)
        for arm_state in agent._arms:
            np.testing.assert_array_equal(arm_state.A, np.eye(6))

    def test_initial_b_vectors_are_zero(self):
        agent = LinUCBAgent(n_features=6)
        for arm_state in agent._arms:
            np.testing.assert_array_equal(arm_state.b, np.zeros(6))

    def test_is_trained_initially_false(self):
        assert LinUCBAgent().is_trained is False

    def test_reward_history_initially_empty(self):
        assert LinUCBAgent().reward_history == []


# ---------------------------------------------------------------------------
# select_action
# ---------------------------------------------------------------------------

class TestSelectAction:
    def test_returns_valid_arm_index(self, context):
        agent = LinUCBAgent(n_features=10)
        action = agent.select_action(context)
        assert isinstance(action, int)
        assert 0 <= action < N_ARMS

    def test_returns_valid_arm_on_batch(self, context_batch):
        agent = LinUCBAgent(n_features=10)
        for ctx in context_batch:
            action = agent.select_action(ctx)
            assert 0 <= action < N_ARMS

    def test_all_arms_can_be_selected(self, rng):
        """
        Each arm can be selected when it has been given the strongest reward
        signal on a given context (α=0 forces pure exploitation).

        For each target arm we create a fresh agent, give that arm a large
        reward on a fixed context, and confirm it becomes the greedy choice.
        """
        ctx = rng.uniform(0.1, 0.9, 10).astype(np.float32)

        for target_arm in range(N_ARMS):
            agent = LinUCBAgent(n_features=10, alpha=0.0)  # pure exploitation
            for _ in range(20):
                agent.update(ctx, target_arm, reward=10.0)
            assert agent.select_action(ctx) == target_arm, (
                f"Expected arm {target_arm} to be selected after receiving "
                "large rewards, but a different arm won."
            )

    def test_deterministic_on_same_context(self, context):
        """Untrained LinUCB is deterministic — same context → same arm."""
        agent = LinUCBAgent(n_features=10)
        a1 = agent.select_action(context)
        a2 = agent.select_action(context)
        assert a1 == a2


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_sets_is_trained(self, context):
        agent = LinUCBAgent(n_features=10)
        agent.update(context, action=0, reward=1.0)
        assert agent.is_trained is True

    def test_update_appends_reward_history(self, context):
        agent = LinUCBAgent(n_features=10)
        agent.update(context, action=1, reward=0.8)
        assert len(agent.reward_history) == 1
        action_logged, reward_logged = agent.reward_history[0]
        assert action_logged == 1
        assert reward_logged == pytest.approx(0.8)

    def test_update_only_modifies_chosen_arm(self, context):
        agent = LinUCBAgent(n_features=10)
        agent.update(context, action=0, reward=1.0)
        # Arm 0 should differ from identity; arms 1–3 must stay at identity
        assert not np.allclose(agent._arms[0].A, np.eye(10))
        for i in range(1, N_ARMS):
            np.testing.assert_array_equal(agent._arms[i].A, np.eye(10))

    def test_update_invalid_action_raises(self, context):
        agent = LinUCBAgent(n_features=10)
        with pytest.raises(ValueError, match="action must be in"):
            agent.update(context, action=N_ARMS, reward=1.0)

    def test_update_negative_action_raises(self, context):
        agent = LinUCBAgent(n_features=10)
        with pytest.raises(ValueError):
            agent.update(context, action=-1, reward=1.0)

    def test_multiple_updates_increase_arm_count(self, context):
        agent = LinUCBAgent(n_features=10)
        for _ in range(5):
            agent.update(context, action=2, reward=0.5)
        assert agent._arms[2].n_updates == 5
        assert agent._arms[0].n_updates == 0


# ---------------------------------------------------------------------------
# get_arm_confidence
# ---------------------------------------------------------------------------

class TestGetArmConfidence:
    def test_shape(self, context):
        agent = LinUCBAgent(n_features=10)
        scores = agent.get_arm_confidence(context)
        assert scores.shape == (N_ARMS,)

    def test_all_equal_when_untrained(self, context):
        """Untrained agent: all arms have identity A and zero b.
        UCB_a = x^T θ_a + α √(x^T A_a⁻¹ x)  →  0 + α ||x|| for all a."""
        agent = LinUCBAgent(n_features=10, alpha=1.0)
        scores = agent.get_arm_confidence(context)
        # All exploration terms are equal (same A_inv = I, same x)
        assert np.allclose(scores, scores[0])

    def test_values_are_finite(self, trained_agent, context):
        scores = trained_agent.get_arm_confidence(context)
        assert np.all(np.isfinite(scores))


# ---------------------------------------------------------------------------
# Exploration: alpha effect
# ---------------------------------------------------------------------------

class TestAlphaEffect:
    def test_high_alpha_prefers_unexplored_arm(self, rng):
        """
        After a single positive-reward update on arm 0:
        - alpha=0  (pure exploitation) → arm 0 wins (has positive θ)
        - alpha=5  (high exploration)  → an *unexplored* arm wins because
          its A_a is still identity (A_a⁻¹ = I) while arm 0's A_a has
          grown (A_a⁻¹ shrinks), so unexplored UCB bonus is larger.
        """
        ctx = rng.uniform(0.1, 0.9, size=10).astype(np.float32)

        agent_exploit = LinUCBAgent(n_features=10, alpha=0.0)
        agent_explore = LinUCBAgent(n_features=10, alpha=5.0)

        # Single positive update on arm 0
        agent_exploit.update(ctx, action=0, reward=2.0)
        agent_explore.update(ctx, action=0, reward=2.0)

        assert agent_exploit.select_action(ctx) == 0, (
            "Pure-exploitation agent (α=0) should pick arm 0 after positive reward"
        )
        assert agent_explore.select_action(ctx) != 0, (
            "High-exploration agent (α=5) should prefer an unexplored arm"
        )

    def test_alpha_zero_is_greedy(self, rng):
        """With α=0, UCB scores equal the ridge-regression predictions only.
        Training arm 2 to have the highest predicted reward makes it always chosen."""
        ctx = rng.uniform(0.1, 0.9, size=10).astype(np.float32)
        agent = LinUCBAgent(n_features=10, alpha=0.0)

        # Give arm 2 many high-reward updates; others get zero
        for _ in range(30):
            agent.update(ctx, action=2, reward=5.0)

        assert agent.select_action(ctx) == 2

    def test_ucb_spread_increases_with_alpha(self, context):
        """Higher alpha → wider spread between the best and worst UCB scores."""
        agent_low  = LinUCBAgent(n_features=10, alpha=0.1)
        agent_high = LinUCBAgent(n_features=10, alpha=5.0)

        spread_low  = np.ptp(agent_low.get_arm_confidence(context))
        spread_high = np.ptp(agent_high.get_arm_confidence(context))
        # Untrained agents: spread is zero (all equal)
        assert spread_low  == pytest.approx(0.0)
        assert spread_high == pytest.approx(0.0)

        # After one update, exploration terms differ between updated and fresh arms
        agent_low.update(context, action=0, reward=1.0)
        agent_high.update(context, action=0, reward=1.0)

        spread_low_after  = np.ptp(agent_low.get_arm_confidence(context))
        spread_high_after = np.ptp(agent_high.get_arm_confidence(context))
        assert spread_high_after > spread_low_after, (
            "Higher alpha should create wider UCB score spread after an update"
        )


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------

class TestRecommend:
    def test_returns_expected_keys(self, trained_agent, context):
        rec = trained_agent.recommend(context, default_prob=0.45)
        required = {"action", "offer_type", "offer_label", "ucb_scores", "confidence", "default_probability"}
        assert required <= set(rec.keys())

    def test_action_is_valid_arm_index(self, trained_agent, context):
        rec = trained_agent.recommend(context, default_prob=0.5)
        assert 0 <= rec["action"] < N_ARMS

    def test_confidence_in_unit_interval(self, trained_agent, context_batch):
        for ctx in context_batch[:10]:
            rec = trained_agent.recommend(ctx, default_prob=0.5)
            assert 0.0 <= rec["confidence"] <= 1.0

    def test_offer_type_matches_action(self, trained_agent, context):
        rec = trained_agent.recommend(context, default_prob=0.5)
        assert rec["offer_type"] == LinUCBArm(rec["action"]).name

    def test_ucb_scores_has_all_arm_keys(self, trained_agent, context):
        rec = trained_agent.recommend(context, default_prob=0.5)
        assert set(rec["ucb_scores"].keys()) == {arm.name for arm in LinUCBArm}

    def test_default_probability_stored(self, trained_agent, context):
        rec = trained_agent.recommend(context, default_prob=0.72)
        assert rec["default_probability"] == pytest.approx(0.72, abs=1e-4)


# ---------------------------------------------------------------------------
# arm_update_counts
# ---------------------------------------------------------------------------

class TestArmUpdateCounts:
    def test_returns_dict_with_all_arms(self, trained_agent):
        counts = trained_agent.arm_update_counts()
        assert set(counts.keys()) == {arm.name for arm in LinUCBArm}

    def test_counts_are_non_negative_integers(self, trained_agent):
        for v in trained_agent.arm_update_counts().values():
            assert isinstance(v, int)
            assert v >= 0

    def test_total_equals_reward_history_length(self, trained_agent):
        total = sum(trained_agent.arm_update_counts().values())
        assert total == len(trained_agent.reward_history)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_save_load_roundtrip_same_action(self, trained_agent, context, tmp_path):
        path = tmp_path / "linucb.pkl"
        trained_agent.save(path)
        loaded = LinUCBAgent.load(path)
        assert trained_agent.select_action(context) == loaded.select_action(context)

    def test_save_load_preserves_ucb_scores(self, trained_agent, context, tmp_path):
        path = tmp_path / "linucb2.pkl"
        trained_agent.save(path)
        loaded = LinUCBAgent.load(path)
        orig   = trained_agent.get_arm_confidence(context)
        loaded_scores = loaded.get_arm_confidence(context)
        np.testing.assert_array_almost_equal(orig, loaded_scores, decimal=8)

    def test_save_load_preserves_arm_counts(self, trained_agent, tmp_path):
        path = tmp_path / "linucb3.pkl"
        trained_agent.save(path)
        loaded = LinUCBAgent.load(path)
        assert trained_agent.arm_update_counts() == loaded.arm_update_counts()

    def test_saved_file_is_valid_pickle(self, trained_agent, tmp_path):
        path = tmp_path / "linucb4.pkl"
        trained_agent.save(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, LinUCBAgent)
