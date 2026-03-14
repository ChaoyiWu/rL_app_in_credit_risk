"""
LinUCB contextual bandit — training and off-policy evaluation.

Usage
-----
    # Requires models/customers.parquet (run train.py first)
    python scripts/train_bandit.py

    # Full options
    python scripts/train_bandit.py \\
        --data-path        models/customers.parquet \\
        --classifier-path  models/default_risk_classifier.pkl \\
        --output-dir       models \\
        --alpha            1.0 \\
        --n-passes         3

Pipeline
--------
1.  Load customer data from Parquet
2.  Load (or train lightweight) XGBoost classifier → default_probs
3.  Extract 10-dim context vectors via extract_rl_state()
4.  Simulate rule-based logging policy → historical (actions, rewards, propensities)
5.  Train LinUCB online  (n_passes sequential sweeps through the data)
6.  Compute IPS weights: LinUCB vs rule-based logging policy
7.  Fit ridge regression reward model for DM / DR estimators
8.  Off-policy evaluation  (rule-based, LinUCB, random)
9.  Print OPE summary table
10. Save LinUCBAgent → output-dir/linucb_agent.pkl
11. Save learning-curve plot → output-dir/plots/linucb_learning_curve.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

from resiliency.data.generator import LABEL_COL
from resiliency.evaluation.ips import effective_sample_size, importance_weights
from resiliency.evaluation.ope import OPEEvaluator
from resiliency.models.classifier import DefaultRiskClassifier
from resiliency.models.linucb import ARM_LABELS, N_ARMS, LinUCBAgent, LinUCBArm
from resiliency.models.rl_agent import OfferType, extract_rl_state


# ---------------------------------------------------------------------------
# LinUCB arm → OfferType mapping and per-arm cost overrides
# ---------------------------------------------------------------------------

_LINUCB_TO_OFFERTYPE: dict[int, OfferType] = {
    LinUCBArm.PAYMENT_PLAN:     OfferType.PAYMENT_PLAN,
    LinUCBArm.SETTLEMENT_30PCT: OfferType.SETTLEMENT_OFFER,
    LinUCBArm.SETTLEMENT_50PCT: OfferType.SETTLEMENT_OFFER,
    LinUCBArm.HARDSHIP_PROGRAM: OfferType.HARDSHIP_PROGRAM,
}

# Settlement cost overrides:
#   SETTLEMENT_30PCT → debtor pays 30 ¢/$  (lender forgives 70 %) → cost 0.55
#   SETTLEMENT_50PCT → debtor pays 50 ¢/$  (lender forgives 50 %) → cost 0.35
_ARM_COST_OVERRIDE: dict[int, float] = {
    LinUCBArm.SETTLEMENT_30PCT: 0.55,
    LinUCBArm.SETTLEMENT_50PCT: 0.35,
}

_BASE_COST: dict[OfferType, float] = {
    OfferType.PAYMENT_PLAN:     0.05,
    OfferType.HARDSHIP_PROGRAM: 0.10,
    OfferType.SETTLEMENT_OFFER: 0.35,
}

_BASE_SATISFACTION: dict[OfferType, float] = {
    OfferType.PAYMENT_PLAN:     0.70,
    OfferType.HARDSHIP_PROGRAM: 0.85,
    OfferType.SETTLEMENT_OFFER: 0.65,
}


# ---------------------------------------------------------------------------
# Domain reward function for 4 LinUCB arms
# ---------------------------------------------------------------------------

def compute_bandit_reward(
    linucb_action: int,
    customer: pd.Series,
    default_prob: float,
) -> float:
    """
    Domain-driven reward for one LinUCB arm selection.

    Uses the same formula as DebtResolutionEnv._compute_reward():

        r = 2 × resolution_prob
          − 1.5 × cost_factor
          + 0.5 × satisfaction
          − 0.5 × (1 − resolution_prob) × default_prob

    Parameters
    ----------
    linucb_action : int
        One of 0–3  (LinUCBArm enum).
    customer : pd.Series
        Customer feature row.
    default_prob : float
        XGBoost predicted 12-month default probability.

    Returns
    -------
    float
    """
    arm   = LinUCBArm(linucb_action)
    offer = _LINUCB_TO_OFFERTYPE[arm]

    p         = float(default_prob)
    delinq    = float(customer.get("months_delinquent",        0))
    requested = bool(customer.get("requested_hardship_program", False))
    severity  = int(customer.get("hardship_severity",           1))

    # Resolution probability per offer type
    if offer is OfferType.PAYMENT_PLAN:
        res = 0.55 + 0.20 * requested - 0.10 * (severity == 2)
    elif offer is OfferType.HARDSHIP_PROGRAM:
        res = 0.60 + 0.15 * requested + 0.05 * (delinq < 3)
    else:  # SETTLEMENT_OFFER
        res = 0.50 + 0.20 * (p > 0.70) - 0.10 * (severity < 1)
    res = float(np.clip(res, 0.05, 0.95))

    cost = _ARM_COST_OVERRIDE.get(linucb_action, _BASE_COST[offer])
    sat  = _BASE_SATISFACTION[offer]

    return float(2.0 * res - 1.5 * cost + 0.5 * sat - 0.5 * (1 - res) * p)


# ---------------------------------------------------------------------------
# Rule-based logging policy
# ---------------------------------------------------------------------------

#  The rule policy is near-deterministic: 0.85 mass on the chosen arm,
#  0.05 each on the other three.  This satisfies the OPE coverage condition
#  (π_0(a|x) > 0 for all a, x).
_RULE_P_CHOSEN = 0.85
_RULE_P_OTHER  = (1.0 - _RULE_P_CHOSEN) / (N_ARMS - 1)


def rule_based_action(customer: pd.Series, default_prob: float) -> int:
    """
    Deterministic credit-resiliency rule hierarchy.

    Priority order
    --------------
    1. Requested hardship OR severity ≥ 2  → Hardship Program
    2. default_prob > 60 %                 → Settlement 30 %  (aggressive)
    3. months_delinquent > 6               → Settlement 50 %  (moderate)
    4. Otherwise                           → Payment Plan
    """
    requested = bool(customer.get("requested_hardship_program", False))
    severity  = int(customer.get("hardship_severity",           1))
    delinq    = float(customer.get("months_delinquent",         0))

    if requested or severity >= 2:
        return int(LinUCBArm.HARDSHIP_PROGRAM)
    if default_prob > 0.60:
        return int(LinUCBArm.SETTLEMENT_30PCT)
    if delinq > 6:
        return int(LinUCBArm.SETTLEMENT_50PCT)
    return int(LinUCBArm.PAYMENT_PLAN)


def rule_action_propensity(chosen_action: int) -> float:  # noqa: ARG001
    """Soft propensity of the rule policy for its chosen action."""
    return _RULE_P_CHOSEN


# ---------------------------------------------------------------------------
# Ridge regression reward model  (used by DM and DR estimators)
# ---------------------------------------------------------------------------

class _RidgeRewardModel:
    """
    Ridge regression mapping  [context | one_hot(action)] → reward.

    Provides the ``(X, actions) → np.ndarray`` interface expected by
    :class:`~resiliency.evaluation.ope.OPEEvaluator`.
    """

    def __init__(self, n_arms: int = N_ARMS, alpha: float = 1.0) -> None:
        self._ridge = Ridge(alpha=alpha)
        self._n_arms = n_arms

    def fit(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> "_RidgeRewardModel":
        self._ridge.fit(self._featurise(contexts, actions), rewards)
        return self

    def __call__(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        return self._ridge.predict(self._featurise(contexts, actions))

    def _featurise(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        one_hot = np.zeros((len(contexts), self._n_arms), dtype=np.float64)
        one_hot[np.arange(len(contexts)), actions] = 1.0
        return np.hstack([contexts.astype(np.float64), one_hot])


# ---------------------------------------------------------------------------
# Propensity helpers
# ---------------------------------------------------------------------------

def linucb_propensity_vector(
    agent: LinUCBAgent,
    contexts: np.ndarray,
    target_actions: np.ndarray,
    epsilon: float = 0.10,
) -> np.ndarray:
    """
    Compute π_LinUCB(target_action | context) for each row.

    Uses an ε-greedy interpretation of the trained agent:
      - greedy arm gets probability  1 − ε + ε/K
      - other arms each get          ε / K

    Parameters
    ----------
    agent : LinUCBAgent
        Trained agent.
    contexts : np.ndarray of shape (n, d)
    target_actions : np.ndarray of shape (n,)
        The actions we want the propensity *of* (e.g. the logging policy's
        choices, when computing IPS weights for OPE).
    epsilon : float
        Exploration floor.  Keeps propensities bounded away from 0.

    Returns
    -------
    np.ndarray of shape (n,)
        π_LinUCB(target_action[i] | contexts[i]).
    """
    p_greedy = 1.0 - epsilon + epsilon / N_ARMS
    p_other  = epsilon / N_ARMS

    propensities = np.empty(len(contexts), dtype=np.float64)
    for i, (ctx, act) in enumerate(zip(contexts, target_actions)):
        greedy_arm = agent.select_action(ctx)
        propensities[i] = p_greedy if (greedy_arm == act) else p_other
    return propensities


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resiliency Intelligence — LinUCB bandit training + OPE"
    )
    parser.add_argument(
        "--data-path",
        default="models/customers.parquet",
        help="Path to customer parquet file (default: models/customers.parquet)",
    )
    parser.add_argument(
        "--classifier-path",
        default="models/default_risk_classifier.pkl",
        help="Path to saved DefaultRiskClassifier (default: models/default_risk_classifier.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save agent and plots (default: models)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="LinUCB exploration coefficient α (default: 1.0)",
    )
    parser.add_argument(
        "--n-passes",
        type=int,
        default=3,
        help="Number of sequential sweeps through the data for online training (default: 3)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir   = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("=== Step 1: Loading customer data ===")
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(
            "Data file not found: {}\n"
            "Run  python scripts/train.py --output-dir {}  first.",
            data_path,
            args.output_dir,
        )
        sys.exit(1)

    df = pd.read_parquet(data_path)
    logger.info("Loaded {:,} customers from {}", len(df), data_path)

    # ------------------------------------------------------------------
    # 2. Load (or train) default-risk classifier → default_probs
    # ------------------------------------------------------------------
    logger.info("=== Step 2: Loading classifier ===")
    clf_path = Path(args.classifier_path)

    if clf_path.exists():
        clf = DefaultRiskClassifier.load(clf_path)
    else:
        # Fallback: train a lightweight classifier on the loaded data
        logger.warning(
            "Classifier not found at {}. Training a lightweight model …", clf_path
        )
        from resiliency.data.generator import GeneratorConfig, CustomerDataGenerator
        feature_cols = [c for c in df.columns if c not in {LABEL_COL, "hardship_severity"}]
        X_all = df[feature_cols]
        y_all = df[LABEL_COL]
        clf = DefaultRiskClassifier(
            xgb_params={
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.10,
                "scale_pos_weight": 3.5,
                "eval_metric": "aucpr",
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",
            },
            calibrate=False,
        )
        clf.fit(X_all, y_all)
        clf.save(clf_path)

    feature_cols = [c for c in df.columns if c not in {LABEL_COL, "hardship_severity"}]
    default_probs: np.ndarray = clf.predict_proba(df[feature_cols])
    logger.info(
        "Default probabilities computed  mean={:.1%}  high-risk(>60%)={:.1%}",
        default_probs.mean(),
        (default_probs > 0.60).mean(),
    )

    # ------------------------------------------------------------------
    # 3. Extract context vectors  (shape: n × 10)
    # ------------------------------------------------------------------
    logger.info("=== Step 3: Extracting context vectors ===")
    n = len(df)
    n_features = 10  # len(RL_STATE_FEATURES)
    contexts = np.stack(
        [extract_rl_state(df.iloc[i]) for i in range(n)],
        axis=0,
    ).astype(np.float64)   # (n, 10)
    logger.info("Context matrix shape: {}", contexts.shape)

    # ------------------------------------------------------------------
    # 4. Simulate rule-based logging policy
    # ------------------------------------------------------------------
    logger.info("=== Step 4: Simulating rule-based logging policy ===")
    log_actions     = np.empty(n, dtype=np.int64)
    log_rewards     = np.empty(n, dtype=np.float64)
    log_propensity  = np.full(n, _RULE_P_CHOSEN)   # same for all (deterministic rule)

    for i in range(n):
        row  = df.iloc[i]
        p    = float(default_probs[i])
        act  = rule_based_action(row, p)
        log_actions[i]    = act
        log_rewards[i]    = compute_bandit_reward(act, row, p)

    logger.info(
        "Logging policy | mean_reward={:.4f} | action distribution: {}",
        log_rewards.mean(),
        {
            LinUCBArm(a).name: int((log_actions == a).sum())
            for a in range(N_ARMS)
        },
    )

    # ------------------------------------------------------------------
    # 5. Train LinUCB online  (n_passes sweeps through the dataset)
    # ------------------------------------------------------------------
    logger.info(
        "=== Step 5: Training LinUCB  (α={}, {} passes) ===",
        args.alpha,
        args.n_passes,
    )
    agent = LinUCBAgent(n_features=n_features, alpha=args.alpha)

    rng = np.random.default_rng(42)
    total_updates = args.n_passes * n
    log_every     = max(total_updates // 10, 1)

    step = 0
    for pass_idx in range(args.n_passes):
        # Shuffle order within each pass
        order = rng.permutation(n)
        for i in order:
            ctx  = contexts[i]
            p    = float(default_probs[i])
            row  = df.iloc[i]

            # Select and observe reward for LinUCB's chosen action
            act    = agent.select_action(ctx)
            reward = compute_bandit_reward(act, row, p)
            agent.update(ctx, act, reward)

            step += 1
            if step % log_every == 0:
                recent = [r for _, r in agent.reward_history[-log_every:]]
                logger.info(
                    "Pass {}/{} | step {:>7,}/{} | avg_reward={:.4f} | arm counts: {}",
                    pass_idx + 1,
                    args.n_passes,
                    step,
                    total_updates,
                    float(np.mean(recent)),
                    agent.arm_update_counts(),
                )

    logger.success(
        "LinUCB training complete — total updates: {:,}", len(agent.reward_history)
    )

    # ------------------------------------------------------------------
    # 6. IPS weights: LinUCB vs rule-based logging policy
    # ------------------------------------------------------------------
    logger.info("=== Step 6: Computing IPS weights ===")

    # π_LinUCB(log_action | context) for each customer
    linucb_propensity = linucb_propensity_vector(
        agent, contexts, log_actions, epsilon=0.10
    )
    # π_random(log_action | context) — uniform over 4 arms
    random_propensity = np.full(n, 1.0 / N_ARMS)

    ips_weights = importance_weights(log_propensity, linucb_propensity)
    ess         = effective_sample_size(log_propensity, linucb_propensity)

    logger.info(
        "IPS diagnostics (LinUCB vs rule-based) | "
        "w_mean={:.3f}  w_max={:.3f}  w_min={:.3f}  ESS={:.0f} ({:.1f}%)",
        ips_weights.mean(),
        ips_weights.max(),
        ips_weights.min(),
        ess,
        100.0 * ess / n,
    )

    # ------------------------------------------------------------------
    # 7. Fit ridge reward model for DM / DR
    # ------------------------------------------------------------------
    logger.info("=== Step 7: Fitting ridge reward model ===")
    reward_model = _RidgeRewardModel(n_arms=N_ARMS)
    reward_model.fit(contexts, log_actions, log_rewards)

    # Quick sanity check: in-sample R²
    predicted = reward_model(contexts, log_actions)
    ss_res = float(np.sum((log_rewards - predicted) ** 2))
    ss_tot = float(np.sum((log_rewards - log_rewards.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    logger.info("Reward model in-sample R² = {:.4f}", r2)

    # ------------------------------------------------------------------
    # 8. Off-policy evaluation
    # ------------------------------------------------------------------
    logger.info("=== Step 8: Off-policy evaluation ===")

    evaluator = OPEEvaluator(
        contexts=contexts,
        actions=log_actions,
        rewards=log_rewards,
        historical_propensity=log_propensity,
        reward_model=reward_model,
    )

    # Greedy actions for each policy (used for DM)
    linucb_greedy  = np.array([agent.select_action(contexts[i]) for i in range(n)])
    random_greedy  = rng.integers(0, N_ARMS, size=n)
    rule_greedy    = log_actions  # rule policy is already greedy

    def _evaluate_policy(
        name: str,
        new_propensity: np.ndarray,
        greedy_actions: np.ndarray,
    ) -> dict:
        dm   = evaluator.direct_method(reward_model, contexts, greedy_actions)
        ips  = evaluator.ips(log_rewards, log_propensity, new_propensity)
        dr   = evaluator.doubly_robust(
            log_rewards, reward_model, contexts, log_actions,
            log_propensity, new_propensity,
        )
        w    = importance_weights(log_propensity, new_propensity)
        ess_ = effective_sample_size(log_propensity, new_propensity)
        return {
            "policy": name,
            "dm":       round(dm,   4),
            "ips":      round(ips,  4),
            "dr":       round(dr,   4),
            "ess":      round(ess_, 1),
            "ess_pct":  round(100.0 * ess_ / n, 1),
            "w_mean":   round(float(w.mean()), 4),
            "w_max":    round(float(w.max()),  4),
        }

    results = [
        _evaluate_policy("Rule-Based (logging)", log_propensity,    rule_greedy),
        _evaluate_policy("LinUCB",               linucb_propensity, linucb_greedy),
        _evaluate_policy("Random",               random_propensity,  random_greedy),
    ]
    ope_df = pd.DataFrame(results).set_index("policy")

    # ------------------------------------------------------------------
    # 9. Print summary table
    # ------------------------------------------------------------------
    logger.info("=== Step 9: OPE summary ===")

    col_labels = {
        "dm":      "DM",
        "ips":     "IPS",
        "dr":      "DR (doubly-robust)",
        "ess":     "ESS",
        "ess_pct": "ESS %",
        "w_mean":  "w̄",
        "w_max":   "w_max",
    }
    print()
    print("=" * 70)
    print("  Off-Policy Evaluation — Estimated Expected Reward per Policy")
    print("=" * 70)
    print(ope_df.rename(columns=col_labels).to_string())
    print()
    print("Estimators: DM = Direct Method · IPS = Importance Sampling")
    print("            DR = Doubly Robust  (preferred — combines DM + IPS)")
    print()

    best_policy = ope_df["dr"].idxmax()
    best_dr     = ope_df.loc[best_policy, "dr"]
    baseline_dr = ope_df.loc["Rule-Based (logging)", "dr"]
    lift        = 100.0 * (best_dr - baseline_dr) / abs(baseline_dr) if baseline_dr != 0 else 0.0

    logger.success(
        "Best policy by DR: '{}' (DR={:.4f}, {:.1f}% lift over rule-based)",
        best_policy,
        best_dr,
        lift,
    )

    # Arm distribution of trained LinUCB
    print("LinUCB arm selection distribution (greedy, post-training):")
    for arm_idx in range(N_ARMS):
        count = int((linucb_greedy == arm_idx).sum())
        pct   = 100.0 * count / n
        print(f"  {LinUCBArm(arm_idx).name:<20s}  {count:>6,} ({pct:>5.1f}%)")
    print()

    # ------------------------------------------------------------------
    # 10. Save LinUCB agent
    # ------------------------------------------------------------------
    logger.info("=== Step 10: Saving LinUCB agent ===")
    agent_path = out_dir / "linucb_agent.pkl"
    agent.save(agent_path)

    # ------------------------------------------------------------------
    # 11. Save learning curve plot
    # ------------------------------------------------------------------
    logger.info("=== Step 11: Saving learning curve plot ===")
    fig = agent.plot_reward_history(window=min(200, len(agent.reward_history) // 10))
    plot_path = plots_dir / "linucb_learning_curve.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info("Learning curve saved → {}", plot_path)

    import matplotlib.pyplot as plt
    plt.close(fig)

    logger.success(
        "LinUCB bandit pipeline complete.\n"
        "  Agent  → {}\n"
        "  Plot   → {}",
        agent_path,
        plot_path,
    )


if __name__ == "__main__":
    main()
