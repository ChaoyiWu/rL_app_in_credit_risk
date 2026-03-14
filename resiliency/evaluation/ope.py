"""
Off-Policy Evaluation (OPE) for debt-resolution offer policies.

Why OPE?
--------
After deploying a new policy (e.g. LinUCB) we cannot cheaply A/B test every
candidate against live customers.  OPE lets us estimate the expected reward of
a *target* policy π_1 using data collected under a *different* logging policy
π_0 — without running π_1 in production.

Three complementary estimators are provided, each trading bias for variance:

Direct Method (DM)
~~~~~~~~~~~~~~~~~~
Fit a reward model r̂(x, a) on logged data.  Estimate policy value as the
average *predicted* reward the target policy would receive:

    V̂_DM = (1/n) Σ r̂(x_i, a_i)

Unbiased when r̂ is a perfect reward model; biased otherwise.
Zero variance from importance weights, but all bias is in the model.

Inverse Propensity Scoring (IPS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Re-weight observed rewards by how much more likely the target policy would
have chosen each action compared with the logging policy:

    w_i   = π_1(a_i | x_i) / π_0(a_i | x_i)
    V̂_IPS = (1/n) Σ w_i r_i

Unbiased (given full coverage), but can have high variance when weights are
large.  See :mod:`resiliency.evaluation.ips` for clipping options.

Doubly Robust (DR)
~~~~~~~~~~~~~~~~~~
Combines DM and IPS so that the estimator is consistent if *either* the
reward model or the propensity model is correctly specified:

    V̂_DR = (1/n) Σ [ r̂(x_i, a_i)  +  w_i (r_i − r̂(x_i, a_i)) ]
             ─────────────────────     ─────────────────────────────
               DM baseline               IPS residual correction

When r̂ is accurate, the residual term is small and variance is low.
When r̂ is misspecified, the IPS correction reduces bias.

Usage
-----
::

    from resiliency.evaluation.ope import OPEEvaluator

    evaluator = OPEEvaluator(
        contexts=X_logged,                 # (n, d) float32 context matrix
        actions=actions_logged,            # (n,)   int    chosen arms
        rewards=rewards_logged,            # (n,)   float  observed rewards
        historical_propensity=pi0,         # (n,)   float  π_0(a_i | x_i)
        reward_model=my_reward_model,      # callable (X, actions) → (n,) rewards
    )

    df = evaluator.compare_policies({
        "LinUCB (α=1.0)": pi1_linucb,
        "Q-Learning":     pi1_qlearn,
        "Random":         pi1_uniform,
    })
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Callable, Optional
from loguru import logger

from resiliency.evaluation.ips import (
    ips_estimate,
    snips_estimate,
    effective_sample_size,
    importance_weights,
)


# Type alias — a reward model is any callable (X, actions) → predicted rewards
RewardModel = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# OPE Evaluator
# ---------------------------------------------------------------------------

class OPEEvaluator:
    """
    Off-Policy Evaluation of debt-resolution policies.

    Wraps three estimators — Direct Method, IPS, and Doubly Robust — and
    exposes a :meth:`compare_policies` helper that produces a summary
    DataFrame comparing any number of candidate policies side by side.

    Parameters
    ----------
    contexts : np.ndarray of shape (n, d)
        Context (feature) matrix from the logged interactions.  Typically
        the output of :func:`resiliency.models.rl_agent.extract_rl_state`
        stacked into a 2-D array.
    actions : np.ndarray of shape (n,)
        Integer arm indices chosen by the logging policy π_0.
    rewards : np.ndarray of shape (n,)
        Scalar rewards observed for each logged interaction.
    historical_propensity : np.ndarray of shape (n,)
        Probability of the *chosen* action under the logging policy π_0,
        i.e.  π_0(actions[i] | contexts[i]).  Must be strictly positive.
    reward_model : callable, optional
        A fitted reward model with signature ``(X, actions) → np.ndarray``.
        Required by :meth:`direct_method` and :meth:`doubly_robust`.
        Can also be passed directly to those methods.

    Examples
    --------
    >>> ev = OPEEvaluator(X, actions, rewards, pi0, reward_model=rm)
    >>> v_dm  = ev.direct_method(rm, X, actions)
    >>> v_ips = ev.ips(rewards, pi0, pi1_linucb)
    >>> v_dr  = ev.doubly_robust(rewards, rm, X, actions, pi0, pi1_linucb)
    >>> df    = ev.compare_policies({"LinUCB": pi1_linucb, "Random": pi1_rand})
    """

    def __init__(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        historical_propensity: np.ndarray,
        reward_model: Optional[RewardModel] = None,
    ) -> None:
        self._contexts = np.asarray(contexts, dtype=np.float64)
        self._actions = np.asarray(actions, dtype=np.int64)
        self._rewards = np.asarray(rewards, dtype=np.float64)
        self._historical_propensity = np.asarray(historical_propensity, dtype=np.float64)
        self._reward_model = reward_model

        n = len(self._rewards)
        if not (
            len(self._contexts) == n
            and len(self._actions) == n
            and len(self._historical_propensity) == n
        ):
            raise ValueError(
                "contexts, actions, rewards, and historical_propensity "
                "must all have the same length."
            )

        logger.info(
            "OPEEvaluator initialised — n={:,}  reward_model={}",
            n,
            type(reward_model).__name__ if reward_model is not None else "None",
        )

    # ------------------------------------------------------------------
    # Direct Method (DM)
    # ------------------------------------------------------------------

    def direct_method(
        self,
        reward_model: RewardModel,
        X: np.ndarray,
        actions: np.ndarray,
    ) -> float:
        """
        Direct Method (DM) policy value estimator.

        Uses a pre-fitted reward model to predict what reward the policy
        would receive for each context.  No propensity scores needed.

        .. math::

            \\hat{V}_{\\text{DM}} = \\frac{1}{n} \\sum_{i=1}^{n}
                \\hat{r}(x_i,\\, a_i)

        Parameters
        ----------
        reward_model : callable
            Fitted reward model with signature
            ``(X: np.ndarray, actions: np.ndarray) → np.ndarray`` returning
            predicted scalar rewards of shape ``(n,)``.
        X : np.ndarray of shape (n, d)
            Context matrix to score.  Pass ``self._contexts`` to use the
            stored logged data, or a different matrix for a target policy.
        actions : np.ndarray of shape (n,)
            Arm indices for which to predict rewards.  These should reflect
            the *target* policy's action selections, not the logging policy.

        Returns
        -------
        float
            Estimated expected reward of the target policy.
        """
        X = np.asarray(X, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.int64)

        predicted_rewards = np.asarray(
            reward_model(X, actions), dtype=np.float64
        )
        if predicted_rewards.shape != (len(X),):
            raise ValueError(
                f"reward_model must return shape (n,), got {predicted_rewards.shape}"
            )

        estimate = float(np.mean(predicted_rewards))
        logger.debug("DM estimate={:.4f}", estimate)
        return estimate

    # ------------------------------------------------------------------
    # Inverse Propensity Scoring (IPS)
    # ------------------------------------------------------------------

    def ips(
        self,
        rewards: np.ndarray,
        historical_propensity: np.ndarray,
        new_propensity: np.ndarray,
        clip: float | None = None,
        self_normalized: bool = False,
    ) -> float:
        """
        Inverse Propensity Scoring (IPS) policy value estimator.

        Delegates to :func:`resiliency.evaluation.ips.ips_estimate` (or
        :func:`~resiliency.evaluation.ips.snips_estimate` when
        ``self_normalized=True``).

        .. math::

            w_i &= \\pi_1(a_i \\mid x_i) \\;/\\; \\pi_0(a_i \\mid x_i) \\\\
            \\hat{V}_{\\text{IPS}} &= \\frac{1}{n} \\sum_i w_i\\, r_i

        Parameters
        ----------
        rewards : np.ndarray of shape (n,)
            Observed rewards from the logged interactions.
        historical_propensity : np.ndarray of shape (n,)
            π_0(a_i | x_i) — probability of the chosen action under the
            logging policy.
        new_propensity : np.ndarray of shape (n,)
            π_1(a_i | x_i) — probability of the same action under the
            target policy being evaluated.
        clip : float, optional
            Hard weight clipping threshold to reduce variance.
        self_normalized : bool
            If True, use SNIPS (divide by Σ w_i) instead of plain IPS.

        Returns
        -------
        float
            Estimated expected reward of the target policy.
        """
        if self_normalized:
            return snips_estimate(rewards, historical_propensity, new_propensity, clip=clip)
        return ips_estimate(rewards, historical_propensity, new_propensity, clip=clip)

    # ------------------------------------------------------------------
    # Doubly Robust (DR)
    # ------------------------------------------------------------------

    def doubly_robust(
        self,
        rewards: np.ndarray,
        reward_model: RewardModel,
        X: np.ndarray,
        actions: np.ndarray,
        historical_propensity: np.ndarray,
        new_propensity: np.ndarray,
        clip: float | None = None,
    ) -> float:
        """
        Doubly Robust (DR) policy value estimator.

        Consistent if *either* the reward model or the propensity model
        is correctly specified.  Combines the low-variance DM baseline with
        an IPS-weighted residual correction:

        .. math::

            \\hat{V}_{\\text{DR}} = \\frac{1}{n} \\sum_{i=1}^{n}
                \\Bigl[
                    \\underbrace{\\hat{r}(x_i, a_i)}_{\\text{DM baseline}}
                    +\\;
                    \\underbrace{w_i \\bigl(r_i - \\hat{r}(x_i, a_i)\\bigr)}_{
                        \\text{IPS residual correction}}
                \\Bigr]

        When the reward model is accurate (small residuals), the correction
        adds little variance.  When the reward model is misspecified, the
        IPS term removes the bias.

        Parameters
        ----------
        rewards : np.ndarray of shape (n,)
            Observed rewards.
        reward_model : callable
            Fitted reward model; see :meth:`direct_method`.
        X : np.ndarray of shape (n, d)
            Context matrix.
        actions : np.ndarray of shape (n,)
            Arm indices (logging policy's choices).
        historical_propensity : np.ndarray of shape (n,)
            π_0(a_i | x_i).
        new_propensity : np.ndarray of shape (n,)
            π_1(a_i | x_i).
        clip : float, optional
            Weight clipping for the IPS residual term.

        Returns
        -------
        float
            Doubly robust estimate of policy value.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.int64)

        # DM baseline: predicted reward for each (context, action) pair
        r_hat = np.asarray(reward_model(X, actions), dtype=np.float64)

        # Importance weights
        w = importance_weights(historical_propensity, new_propensity, clip=clip)

        # DR = DM + IPS-weighted residual
        dr_terms = r_hat + w * (rewards - r_hat)
        estimate = float(np.mean(dr_terms))

        residual_magnitude = float(np.mean(np.abs(rewards - r_hat)))
        logger.debug(
            "DR estimate={:.4f}  |  mean_residual={:.4f}  w_mean={:.3f}",
            estimate, residual_magnitude, w.mean(),
        )
        return estimate

    # ------------------------------------------------------------------
    # Policy comparison table
    # ------------------------------------------------------------------

    def compare_policies(
        self,
        policies: dict[str, np.ndarray],
        clip: float | None = None,
        reward_model: Optional[RewardModel] = None,
    ) -> pd.DataFrame:
        """
        Evaluate multiple target policies and return a comparison DataFrame.

        Runs all three estimators (DM, IPS, DR) for each policy, plus
        diagnostic columns (ESS, weight statistics).

        Parameters
        ----------
        policies : dict[str, np.ndarray]
            Mapping from policy name to a **new_propensity** array of shape
            ``(n,)`` — the probability π_1(a_i | x_i) that the *target*
            policy assigns to the action that was *actually taken* by the
            logging policy.

            Tip: if your policy returns a full probability matrix
            ``P`` of shape ``(n, n_arms)``, slice with
            ``P[np.arange(n), actions]`` before passing it here.

        clip : float, optional
            Weight clipping applied to IPS and the DR residual term.

        reward_model : callable, optional
            Override the reward model stored at construction time.
            Required for DM and DR columns; if neither this argument nor the
            constructor's ``reward_model`` is provided, those columns will
            contain ``NaN``.

        Returns
        -------
        pd.DataFrame
            One row per policy.  Columns:

            - ``dm``    — Direct Method estimate
            - ``ips``   — IPS estimate
            - ``snips`` — Self-Normalised IPS estimate
            - ``dr``    — Doubly Robust estimate
            - ``ess``   — Effective Sample Size (out of n)
            - ``ess_pct`` — ESS as % of n
            - ``w_mean`` — Mean importance weight
            - ``w_max``  — Max importance weight
            - ``w_clip_frac`` — Fraction of weights that would be clipped
              (at the specified ``clip`` value; NaN if ``clip`` is None)
        """
        rm = reward_model or self._reward_model
        n = len(self._rewards)

        rows: list[dict[str, Any]] = []

        for name, new_propensity in policies.items():
            new_propensity = np.asarray(new_propensity, dtype=np.float64)
            row: dict[str, Any] = {"policy": name}

            # ---- IPS-family ------------------------------------------
            row["ips"] = round(
                ips_estimate(
                    self._rewards,
                    self._historical_propensity,
                    new_propensity,
                    clip=clip,
                ),
                4,
            )
            row["snips"] = round(
                snips_estimate(
                    self._rewards,
                    self._historical_propensity,
                    new_propensity,
                    clip=clip,
                ),
                4,
            )

            # ---- Direct Method & DR (require reward model) -----------
            if rm is not None:
                row["dm"] = round(
                    self.direct_method(rm, self._contexts, self._actions),
                    4,
                )
                row["dr"] = round(
                    self.doubly_robust(
                        self._rewards,
                        rm,
                        self._contexts,
                        self._actions,
                        self._historical_propensity,
                        new_propensity,
                        clip=clip,
                    ),
                    4,
                )
            else:
                row["dm"] = float("nan")
                row["dr"] = float("nan")

            # ---- Diagnostics ----------------------------------------
            w = importance_weights(
                self._historical_propensity, new_propensity, clip=None
            )
            ess = effective_sample_size(self._historical_propensity, new_propensity)

            row["ess"] = round(ess, 1)
            row["ess_pct"] = round(100.0 * ess / n, 1)
            row["w_mean"] = round(float(w.mean()), 4)
            row["w_max"] = round(float(w.max()), 4)

            if clip is not None:
                row["w_clip_frac"] = round(float(np.mean(w > clip)), 4)
            else:
                row["w_clip_frac"] = float("nan")

            rows.append(row)
            logger.info(
                "OPE | {:30s} | IPS={:.4f}  DR={:.4f}  ESS={:.0f} ({:.1f}%)",
                name,
                row["ips"],
                row["dr"] if not np.isnan(row["dr"]) else float("nan"),
                ess,
                row["ess_pct"],
            )

        df = pd.DataFrame(rows).set_index("policy")
        # Reorder columns for readability
        col_order = ["dm", "ips", "snips", "dr", "ess", "ess_pct", "w_mean", "w_max", "w_clip_frac"]
        df = df[[c for c in col_order if c in df.columns]]
        return df

    # ------------------------------------------------------------------
    # Convenience: evaluate a single policy with all three estimators
    # ------------------------------------------------------------------

    def evaluate(
        self,
        new_propensity: np.ndarray,
        policy_name: str = "target_policy",
        clip: float | None = None,
        reward_model: Optional[RewardModel] = None,
    ) -> dict[str, Any]:
        """
        Run all three estimators for a single target policy.

        Parameters
        ----------
        new_propensity : np.ndarray of shape (n,)
            π_1(a_i | x_i) for the target policy.
        policy_name : str
            Label used in log output.
        clip : float, optional
        reward_model : callable, optional

        Returns
        -------
        dict
            Keys: ``policy``, ``dm``, ``ips``, ``snips``, ``dr``,
            ``ess``, ``ess_pct``.
        """
        row = self.compare_policies(
            {policy_name: new_propensity},
            clip=clip,
            reward_model=reward_model,
        )
        return {"policy": policy_name, **row.iloc[0].to_dict()}
