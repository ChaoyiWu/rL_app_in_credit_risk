"""
LinUCB contextual bandit for debt resolution offer recommendations.

Algorithm
---------
LinUCB (Disjoint model) — Li et al., 2010 "A Contextual-Bandit Approach to
Personalized News Article Recommendation".

Each arm ``a`` maintains an independent ridge-regression model:

    θ_a  =  A_a⁻¹ b_a

where  A_a ∈ ℝ^{d×d}  and  b_a ∈ ℝ^d  are updated online.

Action selection (Upper Confidence Bound)
-----------------------------------------
The agent picks the arm with the highest UCB score:

    a* = argmax_a  [ x^T θ_a  +  α √(x^T A_a⁻¹ x) ]
         ─────────────────────────────────────────────
         exploitation ↑           exploration ↑

    x  : context vector (d-dimensional)
    α  : exploration coefficient — higher → more exploration

Online update (after observing reward r for arm a)
---------------------------------------------------
    A_a  ←  A_a  +  x x^T
    b_a  ←  b_a  +  r x

This is equivalent to incrementally fitting ridge regression with λ=1
(since A_a starts as the identity matrix I_d).

Arms
----
Four targeted credit-resolution offers, a focused subset of the full
OfferType action space used by QLearningAgent:

    0  PAYMENT_PLAN      — extended repayment terms (18–60 months)
    1  SETTLEMENT_30PCT  — settle at 30 % of outstanding balance
    2  SETTLEMENT_50PCT  — settle at 50 % of outstanding balance
    3  HARDSHIP_PROGRAM  — temporary rate reduction + fee waiver

Context
-------
The expected context vector is the 10-dimensional normalised state produced
by :func:`resiliency.models.rl_agent.extract_rl_state`.  Pass
``n_features=10`` (default) or set it to match any alternative featuriser.
"""
from __future__ import annotations

import numpy as np
import pickle
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any
from loguru import logger


# ---------------------------------------------------------------------------
# Arm enum
# ---------------------------------------------------------------------------

class LinUCBArm(IntEnum):
    PAYMENT_PLAN     = 0
    SETTLEMENT_30PCT = 1
    SETTLEMENT_50PCT = 2
    HARDSHIP_PROGRAM = 3


ARM_LABELS: dict[LinUCBArm, str] = {
    LinUCBArm.PAYMENT_PLAN:     "Payment Plan (Extended Terms)",
    LinUCBArm.SETTLEMENT_30PCT: "Settlement Offer – 30 % of Balance",
    LinUCBArm.SETTLEMENT_50PCT: "Settlement Offer – 50 % of Balance",
    LinUCBArm.HARDSHIP_PROGRAM: "Hardship Program (Rate Reduction)",
}

N_ARMS = len(LinUCBArm)


# ---------------------------------------------------------------------------
# Per-arm ridge-regression state
# ---------------------------------------------------------------------------

@dataclass
class _ArmState:
    """
    Per-arm sufficient statistics for LinUCB (disjoint model).

    Parameters
    ----------
    n_features : int
        Dimension ``d`` of the context vector.

    Attributes
    ----------
    A : np.ndarray of shape (d, d)
        Design matrix, initialised to the identity so the ridge penalty λ=1
        is embedded from the start.
    b : np.ndarray of shape (d,)
        Response accumulator.
    n_updates : int
        Number of online updates applied so far.
    """
    n_features: int
    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    n_updates: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.A = np.eye(self.n_features, dtype=np.float64)
        self.b = np.zeros(self.n_features, dtype=np.float64)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def theta(self) -> np.ndarray:
        """Ridge-regression coefficient vector  θ = A⁻¹ b."""
        return np.linalg.solve(self.A, self.b)

    def A_inv(self) -> np.ndarray:
        """Inverse of the design matrix  A⁻¹."""
        return np.linalg.inv(self.A)

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(self, x: np.ndarray, reward: float) -> None:
        """
        Incorporate one (context, reward) observation.

        Updates
        -------
        A  ←  A  +  x x^T
        b  ←  b  +  reward · x
        """
        x = x.astype(np.float64)
        self.A += np.outer(x, x)
        self.b += reward * x
        self.n_updates += 1


# ---------------------------------------------------------------------------
# LinUCB Agent
# ---------------------------------------------------------------------------

class LinUCBAgent:
    """
    Disjoint LinUCB contextual bandit for debt-resolution offer selection.

    Each of the four arms maintains an independent ridge-regression model.
    Action selection balances exploitation (predicted reward) with exploration
    (uncertainty in the estimate) via the Upper Confidence Bound:

        UCB_a(x)  =  x^T θ_a  +  α √(x^T A_a⁻¹ x)

    Parameters
    ----------
    n_arms : int
        Number of arms.  Defaults to 4 (the :class:`LinUCBArm` enum).
    n_features : int
        Dimension of the context vector.  Use 10 when feeding the output of
        :func:`resiliency.models.rl_agent.extract_rl_state`.
    alpha : float
        Exploration coefficient.  Higher values favour less-visited arms.
        Typical range: 0.1 – 2.0.

    Examples
    --------
    >>> from resiliency.models.rl_agent import extract_rl_state
    >>> agent = LinUCBAgent(n_features=10, alpha=1.0)
    >>> ctx = extract_rl_state(customer_row)          # shape (10,)
    >>> arm = agent.select_action(ctx)
    >>> agent.update(ctx, arm, reward=1.2)
    >>> scores = agent.get_arm_confidence(ctx)        # shape (4,)
    """

    def __init__(
        self,
        n_arms: int = N_ARMS,
        n_features: int = 10,
        alpha: float = 1.0,
    ) -> None:
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha

        self._arms: list[_ArmState] = [
            _ArmState(n_features) for _ in range(n_arms)
        ]
        self.reward_history: list[tuple[int, float]] = []   # (arm, reward)
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Core bandit interface
    # ------------------------------------------------------------------

    def select_action(self, context: np.ndarray) -> int:
        """
        Choose the arm with the highest UCB score.

        UCB score for arm ``a``:

            UCB_a  =  x^T θ_a  +  α √(x^T A_a⁻¹ x)
                      ────────    ────────────────────
                      exploit         explore

        Parameters
        ----------
        context : np.ndarray of shape (n_features,)
            Normalised customer context vector.

        Returns
        -------
        int
            Index of the selected arm (see :class:`LinUCBArm`).
        """
        x = context.astype(np.float64)
        ucb_scores = self._ucb_scores(x)
        return int(np.argmax(ucb_scores))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """
        Update the ridge-regression model for the chosen arm.

        Applies the incremental updates:

            A_a  ←  A_a  +  x x^T
            b_a  ←  b_a  +  reward · x

        Parameters
        ----------
        context : np.ndarray of shape (n_features,)
            Context vector presented when the action was taken.
        action : int
            Arm index that was selected.
        reward : float
            Observed scalar reward.
        """
        if not (0 <= action < self.n_arms):
            raise ValueError(f"action must be in [0, {self.n_arms - 1}], got {action}")

        self._arms[action].update(context, reward)
        self.reward_history.append((action, float(reward)))
        self.is_trained = True

    def get_arm_confidence(self, context: np.ndarray) -> np.ndarray:
        """
        Return the UCB score for every arm given the current context.

        Useful for inspection, visualisation, and ranking alternatives.

        Parameters
        ----------
        context : np.ndarray of shape (n_features,)
            Normalised customer context vector.

        Returns
        -------
        np.ndarray of shape (n_arms,)
            UCB scores — higher means the arm is preferred (or less explored).
        """
        x = context.astype(np.float64)
        return self._ucb_scores(x)

    # ------------------------------------------------------------------
    # Recommendation API  (mirrors QLearningAgent.recommend)
    # ------------------------------------------------------------------

    def recommend(
        self, context: np.ndarray, default_prob: float = 0.5
    ) -> dict[str, Any]:
        """
        Recommend the best debt resolution offer for a single customer.

        Parameters
        ----------
        context : np.ndarray of shape (n_features,)
            Normalised customer context from
            :func:`resiliency.models.rl_agent.extract_rl_state`.
        default_prob : float
            Pre-computed default probability from the XGBoost classifier.

        Returns
        -------
        dict
            Keys: ``action``, ``offer_type``, ``offer_label``,
            ``ucb_scores``, ``confidence``, ``default_probability``.
        """
        ucb_scores = self.get_arm_confidence(context)
        best_action = int(np.argmax(ucb_scores))
        arm = LinUCBArm(best_action)

        # Confidence: softmax spread of UCB scores
        shifted = ucb_scores - ucb_scores.max()
        softmax = np.exp(shifted) / np.exp(shifted).sum()
        confidence = float(softmax[best_action])

        return {
            "action": best_action,
            "offer_type": arm.name,
            "offer_label": ARM_LABELS[arm],
            "ucb_scores": {
                LinUCBArm(i).name: round(float(v), 4)
                for i, v in enumerate(ucb_scores)
            },
            "confidence": round(confidence, 4),
            "default_probability": round(default_prob, 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ucb_scores(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the UCB score for all arms.

        UCB_a  =  x^T θ_a  +  α √(x^T A_a⁻¹ x)

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)  — float64

        Returns
        -------
        np.ndarray of shape (n_arms,)
        """
        scores = np.empty(self.n_arms, dtype=np.float64)
        for i, arm in enumerate(self._arms):
            A_inv = arm.A_inv()
            theta = arm.theta()
            exploit = float(x @ theta)
            explore = float(np.sqrt(x @ A_inv @ x))
            scores[i] = exploit + self.alpha * explore
        return scores

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def arm_update_counts(self) -> dict[str, int]:
        """Return the number of online updates applied to each arm."""
        return {
            LinUCBArm(i).name: arm.n_updates
            for i, arm in enumerate(self._arms)
        }

    def plot_reward_history(
        self, window: int = 100, figsize: tuple[int, int] = (10, 4)
    ):
        """Plot rolling average reward over update steps."""
        import matplotlib.pyplot as plt
        import pandas as pd

        if not self.reward_history:
            raise RuntimeError("No reward history — call update() first.")

        rewards = np.array([r for _, r in self.reward_history])
        rolling = pd.Series(rewards).rolling(window).mean()

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(rewards, alpha=0.2, color="#004977", linewidth=0.5, label="Step reward")
        ax.plot(rolling, color="#D03027", linewidth=2, label=f"Rolling avg (w={window})")
        ax.set_xlabel("Update step")
        ax.set_ylabel("Reward")
        ax.set_title("LinUCB Online Learning Progress", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Serialisation  (consistent with rl_agent.pkl convention)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Persist the agent to disk using pickle.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``models/linucb_agent.pkl``).
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("LinUCB agent saved → {}", path)

    @classmethod
    def load(cls, path: str | Path) -> "LinUCBAgent":
        """
        Load a previously saved agent from disk.

        Parameters
        ----------
        path : str or Path
            Path to the ``.pkl`` file written by :meth:`save`.

        Returns
        -------
        LinUCBAgent
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("LinUCB agent loaded ← {}", path)
        return obj
