"""Unit tests for IPS estimators and OPEEvaluator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resiliency.evaluation.ips import (
    clipped_ips_estimate,
    effective_sample_size,
    importance_weights,
    ips_estimate,
    snips_estimate,
)
from resiliency.evaluation.ope import OPEEvaluator


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

N_SAMPLES = 400
N_ARMS    = 4
N_FEATURES = 6


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def logged_data(rng):
    """
    Synthetic bandit log collected under a uniform logging policy (π_0 = 1/4).

    Returns
    -------
    dict with keys:
        contexts           (N, 6) float64
        actions            (N,)   int64   in [0, 3]
        rewards            (N,)   float64
        historical_propensity (N,)   float64  = 1/4 everywhere
    """
    contexts = rng.uniform(0.0, 1.0, (N_SAMPLES, N_FEATURES))
    actions  = rng.integers(0, N_ARMS, size=N_SAMPLES)
    # Reward is arm-specific + small noise so a reward model can learn something
    arm_means = np.array([0.8, 1.2, 0.5, 1.0])
    rewards   = arm_means[actions] + rng.normal(0, 0.1, size=N_SAMPLES)
    pi0       = np.full(N_SAMPLES, 1.0 / N_ARMS)

    return {
        "contexts":              contexts,
        "actions":               actions,
        "rewards":               rewards,
        "historical_propensity": pi0,
    }


@pytest.fixture(scope="module")
def target_propensity(rng, logged_data) -> np.ndarray:
    """
    Soft target policy π_1: assigns arm-1 double weight, rest share the remainder.
    Propensity = π_1(logged_action | context).
    """
    actions   = logged_data["actions"]
    arm_probs = np.array([0.10, 0.60, 0.15, 0.15])          # sums to 1
    return arm_probs[actions]


@pytest.fixture(scope="module")
def reward_model(logged_data):
    """
    A simple reward model: constant arm-mean prediction.
    Returns a callable (contexts, actions) → predicted_rewards.
    """
    from sklearn.linear_model import Ridge

    X   = logged_data["contexts"]
    a   = logged_data["actions"]
    r   = logged_data["rewards"]

    one_hot = np.zeros((len(a), N_ARMS))
    one_hot[np.arange(len(a)), a] = 1.0
    X_feat = np.hstack([X, one_hot])

    ridge = Ridge(alpha=1.0).fit(X_feat, r)

    def _model(contexts, actions):
        oh = np.zeros((len(actions), N_ARMS))
        oh[np.arange(len(actions)), actions] = 1.0
        return ridge.predict(np.hstack([contexts, oh]))

    return _model


@pytest.fixture(scope="module")
def evaluator(logged_data, reward_model) -> OPEEvaluator:
    return OPEEvaluator(
        contexts=logged_data["contexts"],
        actions=logged_data["actions"],
        rewards=logged_data["rewards"],
        historical_propensity=logged_data["historical_propensity"],
        reward_model=reward_model,
    )


# ---------------------------------------------------------------------------
# importance_weights
# ---------------------------------------------------------------------------

class TestImportanceWeights:
    def test_identical_policies_give_unit_weights(self, logged_data):
        pi0 = logged_data["historical_propensity"]
        w = importance_weights(pi0, pi0.copy())
        np.testing.assert_array_almost_equal(w, np.ones(N_SAMPLES))

    def test_shape_matches_input(self, logged_data, target_propensity):
        pi0 = logged_data["historical_propensity"]
        w = importance_weights(pi0, target_propensity)
        assert w.shape == (N_SAMPLES,)

    def test_non_negative(self, logged_data, target_propensity):
        pi0 = logged_data["historical_propensity"]
        w = importance_weights(pi0, target_propensity)
        assert np.all(w >= 0.0)

    def test_higher_propensity_gives_higher_weight(self):
        pi0 = np.array([0.25, 0.25, 0.25])
        pi1 = np.array([0.10, 0.50, 0.80])
        w   = importance_weights(pi0, pi1)
        assert w[2] > w[1] > w[0]

    def test_zero_denominator_raises(self):
        pi0 = np.array([0.5, 0.0, 0.5])
        pi1 = np.array([0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="non-positive"):
            importance_weights(pi0, pi1)

    def test_negative_denominator_raises(self):
        pi0 = np.array([0.5, -0.1])
        pi1 = np.array([0.5,  0.5])
        with pytest.raises(ValueError):
            importance_weights(pi0, pi1)

    def test_clipping_caps_weights(self, logged_data, target_propensity):
        pi0 = logged_data["historical_propensity"]
        clip = 1.5
        w = importance_weights(pi0, target_propensity, clip=clip)
        assert np.all(w <= clip + 1e-12)

    def test_clipping_does_not_change_small_weights(self):
        pi0 = np.array([0.5, 0.5])
        pi1 = np.array([0.5, 0.5])   # weights = 1.0
        w_noclip = importance_weights(pi0, pi1)
        w_clip   = importance_weights(pi0, pi1, clip=2.0)
        np.testing.assert_array_almost_equal(w_noclip, w_clip)


# ---------------------------------------------------------------------------
# IPS estimators
# ---------------------------------------------------------------------------

class TestIPSEstimators:
    def test_ips_same_policy_equals_empirical_mean(self, logged_data):
        """When π_0 = π_1, IPS weights are all 1 → V̂_IPS = mean(rewards)."""
        pi0 = logged_data["historical_propensity"]
        r   = logged_data["rewards"]
        v   = ips_estimate(r, pi0, pi0.copy())
        assert v == pytest.approx(r.mean(), rel=1e-6)

    def test_snips_same_policy_equals_empirical_mean(self, logged_data):
        pi0 = logged_data["historical_propensity"]
        r   = logged_data["rewards"]
        v   = snips_estimate(r, pi0, pi0.copy())
        assert v == pytest.approx(r.mean(), rel=1e-6)

    def test_snips_bounded_by_reward_range(self, logged_data, target_propensity):
        pi0 = logged_data["historical_propensity"]
        r   = logged_data["rewards"]
        v   = snips_estimate(r, pi0, target_propensity)
        # SNIPS lies in [min(r), max(r)] when weights are non-negative
        assert r.min() - 1e-6 <= v <= r.max() + 1e-6

    def test_clipped_ips_less_than_unclipped_for_high_weight_data(self):
        """Clipping reduces the influence of extreme weights."""
        rng = np.random.default_rng(1)
        pi0 = np.full(200, 0.05)          # very small logging propensities
        pi1 = np.full(200, 0.80)          # much larger target propensities
        r   = rng.normal(1.0, 0.1, 200)  # rewards ~1

        v_unclipped = ips_estimate(r, pi0, pi1, clip=None)
        v_clipped   = clipped_ips_estimate(r, pi0, pi1, clip=5.0)

        # Clipping pulls the estimate toward lower weights → |clipped| ≤ |unclipped|
        assert abs(v_clipped) <= abs(v_unclipped) + 1e-6

    def test_ips_returns_scalar(self, logged_data, target_propensity):
        pi0 = logged_data["historical_propensity"]
        v   = ips_estimate(logged_data["rewards"], pi0, target_propensity)
        assert isinstance(v, float)

    def test_snips_returns_scalar(self, logged_data, target_propensity):
        pi0 = logged_data["historical_propensity"]
        v   = snips_estimate(logged_data["rewards"], pi0, target_propensity)
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Effective Sample Size
# ---------------------------------------------------------------------------

class TestEffectiveSampleSize:
    def test_ess_equals_n_for_identical_policies(self, logged_data):
        """When π_0 = π_1, all weights = 1 → ESS = n."""
        pi0 = logged_data["historical_propensity"]
        ess = effective_sample_size(pi0, pi0.copy())
        assert ess == pytest.approx(N_SAMPLES, rel=1e-6)

    def test_ess_less_than_n_for_different_policies(self, logged_data, target_propensity):
        pi0 = logged_data["historical_propensity"]
        ess = effective_sample_size(pi0, target_propensity)
        assert ess < N_SAMPLES

    def test_ess_at_least_one(self, logged_data, target_propensity):
        """ESS ∈ [1, n] for any non-negative weight vector."""
        pi0 = logged_data["historical_propensity"]
        ess = effective_sample_size(pi0, target_propensity)
        assert ess >= 1.0

    def test_ess_decreases_with_more_extreme_policies(self):
        """The more different π_1 is from π_0, the lower the ESS."""
        pi0     = np.full(N_SAMPLES, 0.25)
        pi1_mild    = np.full(N_SAMPLES, 0.35)
        pi1_extreme = np.full(N_SAMPLES, 0.80)
        ess_mild    = effective_sample_size(pi0, pi1_mild)
        ess_extreme = effective_sample_size(pi0, pi1_extreme)
        assert ess_mild > ess_extreme


# ---------------------------------------------------------------------------
# OPEEvaluator — direct_method
# ---------------------------------------------------------------------------

class TestDirectMethod:
    def test_returns_scalar(self, evaluator, logged_data):
        v = evaluator.direct_method(
            evaluator._reward_model,
            logged_data["contexts"],
            logged_data["actions"],
        )
        assert isinstance(v, float)

    def test_shape_error_raised_for_bad_model(self, evaluator, logged_data):
        """Reward model returning wrong shape should raise ValueError."""
        bad_model = lambda X, a: np.zeros((len(X), 2))
        with pytest.raises(ValueError, match="shape"):
            evaluator.direct_method(bad_model, logged_data["contexts"], logged_data["actions"])

    def test_constant_model_returns_that_constant(self, evaluator, logged_data):
        """A model that always predicts c → DM = c."""
        const_model = lambda X, a: np.full(len(X), 3.14)
        v = evaluator.direct_method(const_model, logged_data["contexts"], logged_data["actions"])
        assert v == pytest.approx(3.14, rel=1e-6)


# ---------------------------------------------------------------------------
# OPEEvaluator — ips
# ---------------------------------------------------------------------------

class TestOPEEvaluatorIPS:
    def test_matches_standalone_ips_function(self, evaluator, logged_data, target_propensity):
        pi0 = logged_data["historical_propensity"]
        r   = logged_data["rewards"]

        v_ope        = evaluator.ips(r, pi0, target_propensity)
        v_standalone = ips_estimate(r, pi0, target_propensity)
        assert v_ope == pytest.approx(v_standalone, rel=1e-10)

    def test_self_normalized_matches_snips(self, evaluator, logged_data, target_propensity):
        pi0 = logged_data["historical_propensity"]
        r   = logged_data["rewards"]

        v_snips = evaluator.ips(r, pi0, target_propensity, self_normalized=True)
        v_ref   = snips_estimate(r, pi0, target_propensity)
        assert v_snips == pytest.approx(v_ref, rel=1e-10)

    def test_returns_scalar(self, evaluator, logged_data, target_propensity):
        pi0 = logged_data["historical_propensity"]
        v   = evaluator.ips(logged_data["rewards"], pi0, target_propensity)
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# OPEEvaluator — doubly_robust
# ---------------------------------------------------------------------------

class TestDoublyRobust:
    def test_returns_scalar(self, evaluator, logged_data, target_propensity):
        pi0 = logged_data["historical_propensity"]
        v   = evaluator.doubly_robust(
            logged_data["rewards"],
            evaluator._reward_model,
            logged_data["contexts"],
            logged_data["actions"],
            pi0,
            target_propensity,
        )
        assert isinstance(v, float)

    def test_dr_equals_dm_when_residuals_are_zero(self, evaluator, logged_data, target_propensity):
        """
        With a perfect reward model (r̂ = r exactly), the IPS residual term
        vanishes:  V̂_DR = E[r̂] + E[w(r − r̂)] = DM + 0 = DM.
        """
        r   = logged_data["rewards"]
        pi0 = logged_data["historical_propensity"]
        X   = logged_data["contexts"]
        a   = logged_data["actions"]

        # Perfect model: predict exact observed rewards
        perfect_model = lambda X, acts: r.copy()

        dm = evaluator.direct_method(perfect_model, X, a)
        dr = evaluator.doubly_robust(r, perfect_model, X, a, pi0, target_propensity)
        assert dr == pytest.approx(dm, rel=1e-6)

    def test_dr_equals_ips_when_model_predicts_zero(self, evaluator, logged_data, target_propensity):
        """
        With a zero reward model (r̂ = 0), the DM baseline is 0 and the full
        IPS correction carries the estimate:  V̂_DR = 0 + E[w(r − 0)] = V̂_IPS.
        """
        r   = logged_data["rewards"]
        pi0 = logged_data["historical_propensity"]
        X   = logged_data["contexts"]
        a   = logged_data["actions"]

        zero_model = lambda X, acts: np.zeros(len(X))

        v_ips = ips_estimate(r, pi0, target_propensity)
        v_dr  = evaluator.doubly_robust(r, zero_model, X, a, pi0, target_propensity)
        assert v_dr == pytest.approx(v_ips, rel=1e-6)

    def test_dr_between_dm_and_ips_with_accurate_model(self, evaluator, logged_data):
        """
        When the reward model is accurate (residuals ≈ 0) and policies are
        similar (IPS weights ≈ 1), DR interpolates between DM and IPS.
        We verify this by using identical policies so IPS = DM = empirical mean,
        and confirming DR is in the same neighbourhood.
        """
        pi0 = logged_data["historical_propensity"]
        r   = logged_data["rewards"]
        X   = logged_data["contexts"]
        a   = logged_data["actions"]

        # With identical policies, IPS weights = 1 → IPS = mean(r)
        v_ips = ips_estimate(r, pi0, pi0.copy())
        v_dm  = evaluator.direct_method(evaluator._reward_model, X, a)
        v_dr  = evaluator.doubly_robust(r, evaluator._reward_model, X, a, pi0, pi0.copy())

        lo, hi = min(v_dm, v_ips), max(v_dm, v_ips)
        tolerance = 0.05 * max(abs(hi - lo), 0.1)   # 5% slack for finite-sample noise
        assert lo - tolerance <= v_dr <= hi + tolerance, (
            f"DR={v_dr:.4f} not between DM={v_dm:.4f} and IPS={v_ips:.4f}"
        )


# ---------------------------------------------------------------------------
# OPEEvaluator — compare_policies
# ---------------------------------------------------------------------------

class TestComparePolicies:
    @pytest.fixture(scope="class")
    def ope_table(self, evaluator, logged_data, target_propensity):
        pi0 = logged_data["historical_propensity"]
        random_pi = np.full(N_SAMPLES, 1.0 / N_ARMS)
        return evaluator.compare_policies(
            {
                "Logging (rule)": pi0.copy(),
                "Target":         target_propensity,
                "Random":         random_pi,
            }
        )

    def test_returns_dataframe(self, ope_table):
        assert isinstance(ope_table, pd.DataFrame)

    def test_row_count_equals_policy_count(self, ope_table):
        assert len(ope_table) == 3

    def test_index_matches_policy_names(self, ope_table):
        assert set(ope_table.index) == {"Logging (rule)", "Target", "Random"}

    def test_required_columns_present(self, ope_table):
        required = {"ips", "snips", "ess", "ess_pct", "w_mean", "w_max"}
        assert required <= set(ope_table.columns)

    def test_dm_and_dr_columns_present_with_reward_model(self, ope_table):
        assert "dm" in ope_table.columns
        assert "dr" in ope_table.columns

    def test_ess_pct_in_range(self, ope_table):
        assert (ope_table["ess_pct"] >= 0).all()
        assert (ope_table["ess_pct"] <= 100.0 + 1e-6).all()

    def test_logging_policy_ess_pct_is_100(self, ope_table):
        """Logging policy vs itself: all IPS weights = 1 → ESS = n → 100 %."""
        assert ope_table.loc["Logging (rule)", "ess_pct"] == pytest.approx(100.0, rel=1e-4)

    def test_w_mean_logging_policy_is_one(self, ope_table):
        assert ope_table.loc["Logging (rule)", "w_mean"] == pytest.approx(1.0, rel=1e-6)

    def test_no_reward_model_gives_nan_dm(self, logged_data, target_propensity):
        """Without a reward model, dm and dr columns should be NaN."""
        pi0 = logged_data["historical_propensity"]
        ev = OPEEvaluator(
            contexts=logged_data["contexts"],
            actions=logged_data["actions"],
            rewards=logged_data["rewards"],
            historical_propensity=pi0,
            reward_model=None,
        )
        df = ev.compare_policies({"Target": target_propensity})
        assert np.isnan(df.loc["Target", "dm"])
        assert np.isnan(df.loc["Target", "dr"])

    def test_compare_with_clip_adds_w_clip_frac_column(
        self, evaluator, logged_data, target_propensity
    ):
        pi0 = logged_data["historical_propensity"]
        df = evaluator.compare_policies(
            {"T": target_propensity}, clip=2.0
        )
        assert "w_clip_frac" in df.columns
        assert 0.0 <= df.loc["T", "w_clip_frac"] <= 1.0

    def test_compare_without_clip_w_clip_frac_is_nan(
        self, evaluator, logged_data, target_propensity
    ):
        df = evaluator.compare_policies({"T": target_propensity}, clip=None)
        assert np.isnan(df.loc["T", "w_clip_frac"])


# ---------------------------------------------------------------------------
# OPEEvaluator — construction validation
# ---------------------------------------------------------------------------

class TestOPEEvaluatorInit:
    def test_mismatched_lengths_raise(self, logged_data):
        with pytest.raises(ValueError, match="same length"):
            OPEEvaluator(
                contexts=logged_data["contexts"][:-1],   # one fewer row
                actions=logged_data["actions"],
                rewards=logged_data["rewards"],
                historical_propensity=logged_data["historical_propensity"],
            )

    def test_reward_model_optional(self, logged_data):
        ev = OPEEvaluator(
            contexts=logged_data["contexts"],
            actions=logged_data["actions"],
            rewards=logged_data["rewards"],
            historical_propensity=logged_data["historical_propensity"],
        )
        assert ev._reward_model is None

    def test_evaluate_convenience_returns_dict(self, evaluator, target_propensity):
        result = evaluator.evaluate(target_propensity, "test_policy")
        assert isinstance(result, dict)
        assert result["policy"] == "test_policy"
        assert "ips" in result and "dr" in result
