"""
End-to-end training pipeline.

Usage
-----
    python scripts/train.py [--n-samples 10000] [--n-rl-episodes 10000] [--output-dir models]

Steps
-----
1. Generate synthetic customer dataset
2. Fit + evaluate the XGBoost default risk classifier
3. Train the Q-learning RL agent using classifier-predicted probabilities
4. Save both models to the specified output directory
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the project root is on sys.path when run directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger
from resiliency.data.generator import CustomerDataGenerator, GeneratorConfig, LABEL_COL
from resiliency.evaluation.metrics import (
    plot_roc_curve,
    plot_confusion_matrix,
    classification_report_df,
    plot_feature_importance,
    business_metrics,
)
from resiliency.models.classifier import DefaultRiskClassifier
from resiliency.models.rl_agent import QLearningAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resiliency Intelligence — training pipeline")
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--n-rl-episodes", type=int, default=10_000)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--save-plots", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    if args.save_plots:
        plots_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    logger.info("=== Step 1: Generating synthetic dataset ===")
    gen = CustomerDataGenerator(GeneratorConfig(n_samples=args.n_samples))
    df = gen.generate()
    train_df, test_df = gen.train_test_split(df)

    X_train = train_df.drop(columns=[LABEL_COL, "hardship_severity"])
    y_train = train_df[LABEL_COL]
    X_test = test_df.drop(columns=[LABEL_COL, "hardship_severity"])
    y_test = test_df[LABEL_COL]

    # Save dataset
    df.to_parquet(out_dir / "customers.parquet", index=False)
    logger.info("Dataset saved → {}", out_dir / "customers.parquet")

    # ------------------------------------------------------------------
    # 2. Train classifier
    # ------------------------------------------------------------------
    logger.info("=== Step 2: Training XGBoost default risk classifier ===")
    clf = DefaultRiskClassifier(calibrate=True)
    clf.fit(X_train, y_train, eval_set=(X_test, y_test))

    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)

    # -- Evaluation reports
    biz = business_metrics(y_test.values, y_prob, y_pred)
    logger.info("\n{}", biz.T.to_string())

    report = classification_report_df(y_test.values, y_pred)
    logger.info("\n{}", report.to_string())

    if args.save_plots:
        plot_roc_curve(y_test.values, y_prob).savefig(plots_dir / "roc_curve.png", dpi=150)
        plot_confusion_matrix(y_test.values, y_pred).savefig(
            plots_dir / "confusion_matrix.png", dpi=150
        )
        plot_feature_importance(
            clf.feature_names_, clf.feature_importances_
        ).savefig(plots_dir / "feature_importance.png", dpi=150)
        logger.info("Plots saved → {}", plots_dir)

    clf.save(out_dir / "default_risk_classifier.pkl")

    # ------------------------------------------------------------------
    # 3. Train RL agent
    # ------------------------------------------------------------------
    logger.info("=== Step 3: Training Q-learning RL agent ===")

    # Use full dataset probabilities for training the RL agent
    all_features = df.drop(columns=[LABEL_COL])
    # Generate probabilities for all rows
    all_X = all_features.drop(columns=["hardship_severity"])
    all_probs = clf.predict_proba(all_X)

    # RL agent trains on full dataset with default probs as context
    rl_train_df = df.copy()  # agent uses hardship_severity column
    agent = QLearningAgent(epsilon_decay=0.9995)
    agent.train(rl_train_df, default_probs=all_probs, n_episodes=args.n_rl_episodes)

    if args.save_plots:
        agent.plot_reward_history().savefig(plots_dir / "rl_reward_history.png", dpi=150)

    agent.save(out_dir / "rl_agent.pkl")

    # ------------------------------------------------------------------
    # 4. Demo recommendation
    # ------------------------------------------------------------------
    logger.info("=== Step 4: Demo recommendation on first test customer ===")
    sample = test_df.iloc[0].to_dict()
    p = float(clf.predict_proba(pd.DataFrame([sample]).drop(columns=[LABEL_COL, "hardship_severity"]))[0])
    rec = agent.recommend(sample, default_prob=p)
    logger.success(
        "Customer default_prob={:.1%} → Recommended: {} (confidence={:.1%})",
        p,
        rec["offer_label"],
        rec["confidence"],
    )

    logger.success("Training pipeline complete. Models saved to: {}", out_dir)


if __name__ == "__main__":
    main()
