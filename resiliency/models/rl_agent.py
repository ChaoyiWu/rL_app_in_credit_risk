"""
Reinforcement Learning agent for debt resolution offer recommendations.

Architecture
------------
- Environment : ``DebtResolutionEnv`` — a Gymnasium-compatible environment
  where each episode is one customer interaction.  The agent selects an
  offer from a discrete action space; the environment returns a reward
  based on the expected resolution success and cost-to-serve.

- Agent : ``QLearningAgent`` — tabular Q-learning with discretised state
  bins.  Suitable for training on the synthetic dataset without GPU.

- Stable-Baselines3 extension : ``train_ppo_agent`` — drops in a PPO agent
  from stable-baselines3 for continuous-state training (optional import).

Action space (OfferType)
------------------------
0  NO_ACTION            — do nothing, monitor only
1  PAYMENT_PLAN         — extend repayment term (18–60 months)
2  HARDSHIP_PROGRAM     — temporary interest-rate reduction + fee waiver
3  SETTLEMENT_OFFER     — accept reduced lump-sum (40–70% of balance)
4  SKIP_PAYMENT         — allow one cycle skip, report deferred
5  CREDIT_COUNSELING    — refer to non-profit credit counselling service
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional
from loguru import logger

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False
    logger.warning("gymnasium not installed — DebtResolutionEnv will be unavailable")


# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------

class OfferType(IntEnum):
    NO_ACTION = 0
    PAYMENT_PLAN = 1
    HARDSHIP_PROGRAM = 2
    SETTLEMENT_OFFER = 3
    SKIP_PAYMENT = 4
    CREDIT_COUNSELING = 5


OFFER_LABELS = {
    OfferType.NO_ACTION: "No Action",
    OfferType.PAYMENT_PLAN: "Payment Plan (Extended Terms)",
    OfferType.HARDSHIP_PROGRAM: "Hardship Program (Rate Reduction)",
    OfferType.SETTLEMENT_OFFER: "Settlement Offer (Reduced Balance)",
    OfferType.SKIP_PAYMENT: "Skip Payment (Deferment)",
    OfferType.CREDIT_COUNSELING: "Credit Counseling Referral",
}

N_ACTIONS = len(OfferType)


# ---------------------------------------------------------------------------
# State feature extraction
# ---------------------------------------------------------------------------

# Features used to construct the RL state vector (subset of full features)
RL_STATE_FEATURES = [
    "credit_score",
    "credit_utilization_pct",
    "months_delinquent",
    "debt_to_income_ratio",
    "num_missed_payments_12m",
    "has_bankruptcy",
    "requested_hardship_program",
    "hardship_severity",        # 0/1/2
    "annual_income",
    "min_payment_ratio",
]

# Bin edges for state discretisation (used by Q-learning)
_BIN_EDGES: dict[str, list[float]] = {
    "credit_score":             [0, 500, 580, 650, 720, 1000],
    "credit_utilization_pct":   [0.0, 0.30, 0.60, 0.80, 1.01],
    "months_delinquent":        [-1, 0, 2, 6, 12, 25],
    "debt_to_income_ratio":     [0, 0.30, 0.60, 1.00, 1.50, 6.0],
    "num_missed_payments_12m":  [-1, 0, 2, 5, 9, 13],
    "has_bankruptcy":           [-0.5, 0.5, 1.5],
    "requested_hardship_program": [-0.5, 0.5, 1.5],
    "hardship_severity":        [-0.5, 0.5, 1.5, 2.5],
    "annual_income":            [0, 25_000, 40_000, 65_000, 100_000, 300_000],
    "min_payment_ratio":        [0, 0.5, 0.8, 1.2, 4.0],
}


def extract_rl_state(customer: dict | pd.Series) -> np.ndarray:
    """
    Extract and normalise a continuous state vector for a single customer.

    Parameters
    ----------
    customer : dict or pd.Series
        Customer feature row.

    Returns
    -------
    np.ndarray of shape (len(RL_STATE_FEATURES),)
        Normalised float32 vector.
    """
    raw = np.array([float(customer.get(f, 0)) for f in RL_STATE_FEATURES], dtype=np.float32)
    # Simple min-max scale using domain knowledge
    scale = np.array([850, 1.0, 24, 5.0, 12, 1, 1, 2, 300_000, 3.0], dtype=np.float32)
    return np.clip(raw / (scale + 1e-9), 0.0, 1.0)


def discretise_state(customer: dict | pd.Series) -> tuple[int, ...]:
    """
    Map a customer's continuous features to a discrete state tuple.

    Used exclusively by the tabular Q-learning agent.
    """
    bins = []
    for feat in RL_STATE_FEATURES:
        val = float(customer.get(feat, 0))
        edges = _BIN_EDGES[feat]
        bucket = int(np.searchsorted(edges, val, side="right")) - 1
        bucket = max(0, min(bucket, len(edges) - 2))
        bins.append(bucket)
    return tuple(bins)


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------

if _GYM_AVAILABLE:

    class DebtResolutionEnv(gym.Env):
        """
        Single-step debt resolution environment.

        Each episode represents one customer interaction.  The agent
        chooses an offer; the environment returns an immediate reward.

        Observation space
        -----------------
        Box(len(RL_STATE_FEATURES),) — normalised float32 features.

        Action space
        ------------
        Discrete(6) — one of the six OfferType values.

        Reward function
        ---------------
        The reward combines:
        - ``resolution_rate``  : probability the customer resolves successfully
        - ``cost_factor``      : intervention cost as fraction of balance
        - ``customer_satisfaction``: proxy for NPS impact
        """

        metadata = {"render_modes": ["human"]}

        def __init__(
            self,
            customer_df: pd.DataFrame,
            default_probs: Optional[np.ndarray] = None,
            random_seed: int = 42,
        ) -> None:
            """
            Parameters
            ----------
            customer_df : pd.DataFrame
                Dataset of customers. Each episode samples one row.
            default_probs : np.ndarray, optional
                Pre-computed default probabilities from the XGBoost classifier.
            random_seed : int
            """
            super().__init__()
            self._df = customer_df.reset_index(drop=True)
            self._default_probs = (
                default_probs
                if default_probs is not None
                else np.full(len(customer_df), 0.5)
            )
            self._rng = np.random.default_rng(random_seed)

            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(RL_STATE_FEATURES),),
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(N_ACTIONS)

            self._current_customer: Optional[pd.Series] = None
            self._current_default_prob: float = 0.5
            self._episode_count: int = 0

        # ---- gym interface -----------------------------------------------

        def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
        ) -> tuple[np.ndarray, dict]:
            super().reset(seed=seed)
            idx = self._rng.integers(0, len(self._df))
            self._current_customer = self._df.iloc[idx]
            self._current_default_prob = float(self._default_probs[idx])
            self._episode_count += 1
            obs = extract_rl_state(self._current_customer)
            return obs, {"customer_idx": int(idx)}

        def step(
            self, action: int
        ) -> tuple[np.ndarray, float, bool, bool, dict]:
            """
            Apply offer ``action`` to the current customer.

            Returns
            -------
            observation : np.ndarray
            reward : float
            terminated : bool  — always True (single-step episode)
            truncated : bool
            info : dict
            """
            offer = OfferType(action)
            reward, info = self._compute_reward(offer)
            obs = extract_rl_state(self._current_customer)
            return obs, reward, True, False, info

        def render(self) -> None:
            if self._current_customer is not None:
                c = self._current_customer
                print(
                    f"Customer | score={c.get('credit_score', 0):.0f} "
                    f"| dti={c.get('debt_to_income_ratio', 0):.2f} "
                    f"| delinq_months={c.get('months_delinquent', 0):.0f} "
                    f"| default_prob={self._current_default_prob:.2%}"
                )

        # ---- reward engineering ------------------------------------------

        def _compute_reward(self, offer: OfferType) -> tuple[float, dict]:
            """
            Domain-driven reward function.

            Returns
            -------
            reward : float   in range [-1, +3]
            info : dict
            """
            p_default = self._current_default_prob
            c = self._current_customer
            dti = float(c.get("debt_to_income_ratio", 0.5))
            delinq = float(c.get("months_delinquent", 0))
            requested = bool(c.get("requested_hardship_program", False))
            severity = int(c.get("hardship_severity", 1))

            # --- resolution probability per offer -------------------------
            # Higher default risk + harder hardship → certain offers work better
            resolution_rates = {
                OfferType.NO_ACTION: max(0.05, 0.40 - 0.5 * p_default),
                OfferType.PAYMENT_PLAN: (
                    0.55 + 0.20 * requested - 0.10 * (severity == 2)
                ),
                OfferType.HARDSHIP_PROGRAM: (
                    0.60 + 0.15 * requested + 0.05 * (delinq < 3)
                ),
                OfferType.SETTLEMENT_OFFER: (
                    0.50 + 0.20 * (p_default > 0.70) - 0.10 * (severity < 1)
                ),
                OfferType.SKIP_PAYMENT: (
                    0.45 - 0.15 * (delinq > 6) + 0.10 * (severity < 2)
                ),
                OfferType.CREDIT_COUNSELING: (
                    0.40 + 0.10 * (dti > 1.0)
                ),
            }
            resolution_prob = float(np.clip(resolution_rates[offer], 0.05, 0.95))

            # --- cost factors (lower cost = better for Capital One) --------
            cost_factors = {
                OfferType.NO_ACTION: 0.0,
                OfferType.PAYMENT_PLAN: 0.05,
                OfferType.HARDSHIP_PROGRAM: 0.10,
                OfferType.SETTLEMENT_OFFER: 0.35,   # forgiven balance
                OfferType.SKIP_PAYMENT: 0.03,
                OfferType.CREDIT_COUNSELING: 0.02,
            }
            cost = cost_factors[offer]

            # --- customer satisfaction ------------------------------------
            satisfaction = {
                OfferType.NO_ACTION: 0.20,
                OfferType.PAYMENT_PLAN: 0.70,
                OfferType.HARDSHIP_PROGRAM: 0.85,
                OfferType.SETTLEMENT_OFFER: 0.65,
                OfferType.SKIP_PAYMENT: 0.75,
                OfferType.CREDIT_COUNSELING: 0.60,
            }[offer]

            # --- composite reward ----------------------------------------
            reward = (
                2.0 * resolution_prob
                - 1.5 * cost
                + 0.5 * satisfaction
                - 0.5 * (1 - resolution_prob) * p_default  # penalty for missed defaults
            )

            info = {
                "offer": offer.name,
                "resolution_probability": resolution_prob,
                "cost_factor": cost,
                "satisfaction": satisfaction,
                "default_probability": p_default,
            }
            return float(reward), info


# ---------------------------------------------------------------------------
# Tabular Q-Learning Agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-learning agent for the debt resolution environment.

    Uses a dictionary Q-table with discretised states so it works without
    a GPU and trains quickly on the synthetic dataset.

    Parameters
    ----------
    n_actions : int
        Number of discrete actions.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor (not used heavily in single-step episodes).
    epsilon_start : float
        Initial exploration probability.
    epsilon_min : float
        Minimum exploration probability after decay.
    epsilon_decay : float
        Multiplicative decay applied each episode.

    Examples
    --------
    >>> agent = QLearningAgent()
    >>> agent.train(env, n_episodes=5000)
    >>> action = agent.act(state_tuple)
    """

    def __init__(
        self,
        n_actions: int = N_ACTIONS,
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self._q: dict[tuple, np.ndarray] = {}
        self.reward_history: list[float] = []
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Q-table helpers
    # ------------------------------------------------------------------

    def _get_q(self, state: tuple) -> np.ndarray:
        if state not in self._q:
            self._q[state] = np.zeros(self.n_actions, dtype=np.float32)
        return self._q[state]

    def act(self, state: tuple, greedy: bool = False) -> int:
        """
        Select an action via epsilon-greedy policy.

        Parameters
        ----------
        state : tuple
            Discretised state from ``discretise_state()``.
        greedy : bool
            If True, always choose the greedy action (inference mode).

        Returns
        -------
        int
            Action index.
        """
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self._get_q(state)))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        customer_df: pd.DataFrame,
        default_probs: Optional[np.ndarray] = None,
        n_episodes: int = 10_000,
        log_every: int = 1_000,
    ) -> "QLearningAgent":
        """
        Run Q-learning training loop.

        Parameters
        ----------
        customer_df : pd.DataFrame
            Customer dataset used to sample episodes.
        default_probs : np.ndarray, optional
            XGBoost predicted probabilities for each customer.
        n_episodes : int
            Total training episodes.
        log_every : int
            Log average reward every N episodes.

        Returns
        -------
        QLearningAgent
            self
        """
        if not _GYM_AVAILABLE:
            raise ImportError("gymnasium is required for training. Run: pip install gymnasium")

        env = DebtResolutionEnv(customer_df, default_probs)
        rng = np.random.default_rng(42)
        episode_rewards = []

        logger.info("Starting Q-learning training for {} episodes", n_episodes)

        for ep in range(n_episodes):
            # Sample a customer
            idx = int(rng.integers(0, len(customer_df)))
            customer = customer_df.iloc[idx]
            state = discretise_state(customer)

            # Single-step episode
            env._current_customer = customer                        # noqa: SLF001
            env._current_default_prob = (                           # noqa: SLF001
                float(default_probs[idx]) if default_probs is not None else 0.5
            )
            action = self.act(state)
            reward, _ = env._compute_reward(OfferType(action))  # noqa: SLF001

            # Q-update (next state = terminal → Q_next = 0)
            q_vals = self._get_q(state)
            q_vals[action] += self.alpha * (reward - q_vals[action])

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_rewards.append(reward)

            if (ep + 1) % log_every == 0:
                avg_r = float(np.mean(episode_rewards[-log_every:]))
                logger.info(
                    "Episode {:>6,} | ε={:.4f} | avg_reward={:.4f} | Q-states={}",
                    ep + 1,
                    self.epsilon,
                    avg_r,
                    len(self._q),
                )

        self.reward_history = episode_rewards
        self.is_trained = True
        logger.success("Q-learning training complete — total Q-states: {}", len(self._q))
        return self

    # ------------------------------------------------------------------
    # Recommendation API
    # ------------------------------------------------------------------

    def recommend(
        self, customer: dict | pd.Series, default_prob: float = 0.5
    ) -> dict[str, Any]:
        """
        Recommend the best debt resolution offer for a single customer.

        Parameters
        ----------
        customer : dict or pd.Series
            Customer feature row.
        default_prob : float
            Pre-computed default probability from the XGBoost classifier.

        Returns
        -------
        dict
            Keys: ``action``, ``offer_type``, ``offer_label``,
            ``q_values``, ``confidence``.
        """
        if not self.is_trained:
            raise RuntimeError("Agent not trained. Call train() first.")

        state = discretise_state(customer)
        q_vals = self._get_q(state).copy()
        best_action = int(np.argmax(q_vals))
        offer = OfferType(best_action)

        # Confidence: soft-max spread of top action vs. mean
        exp_q = np.exp(q_vals - q_vals.max())
        softmax = exp_q / exp_q.sum()
        confidence = float(softmax[best_action])

        return {
            "action": best_action,
            "offer_type": offer.name,
            "offer_label": OFFER_LABELS[offer],
            "q_values": {OfferType(i).name: round(float(v), 4) for i, v in enumerate(q_vals)},
            "confidence": round(confidence, 4),
            "default_probability": round(default_prob, 4),
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("RL agent saved → {}", path)

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("RL agent loaded ← {}", path)
        return obj

    def plot_reward_history(
        self, window: int = 200, figsize: tuple[int, int] = (10, 4)
    ):
        """Plot rolling average reward over training episodes."""
        import matplotlib.pyplot as plt

        rewards = np.array(self.reward_history)
        rolling = pd.Series(rewards).rolling(window).mean()

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(rewards, alpha=0.2, color="#004977", linewidth=0.5, label="Episode reward")
        ax.plot(rolling, color="#D03027", linewidth=2, label=f"Rolling avg (w={window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Q-Learning Training Progress", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Stable-Baselines3 PPO (optional — requires pip install stable-baselines3)
# ---------------------------------------------------------------------------

def train_ppo_agent(
    customer_df: pd.DataFrame,
    default_probs: Optional[np.ndarray] = None,
    total_timesteps: int = 50_000,
    model_path: str = "models/ppo_debt_resolution",
):
    """
    Train a PPO agent using Stable-Baselines3 on the DebtResolutionEnv.

    Requires ``stable-baselines3`` to be installed.

    Parameters
    ----------
    customer_df : pd.DataFrame
    default_probs : np.ndarray, optional
    total_timesteps : int
    model_path : str
        Path prefix to save the trained PPO model.

    Returns
    -------
    stable_baselines3.PPO
        Trained PPO model.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError as e:
        raise ImportError(
            "stable-baselines3 is required. Run: pip install stable-baselines3"
        ) from e

    env = DebtResolutionEnv(customer_df, default_probs)
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.95,
        seed=42,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    logger.success("PPO model saved → {}", model_path)
    return model
