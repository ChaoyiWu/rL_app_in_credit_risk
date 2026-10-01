"""
Train the synthetic-data default-risk model.

Usage
-----
    python scripts/train.py [--n-samples 10000] [--output-dir models]

Steps
-----
1. Generate a synthetic customer dataset
2. Split the data into training and test samples
3. Fit and evaluate the XGBoost default-risk classifier
4. Save the dataset, classifier, and evaluation plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

from resiliency.data.generator import CustomerDataGenerator, GeneratorConfig, LABEL_COL
from resiliency.evaluation.metrics import (
    business_metrics,
    classification_report_df,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from resiliency.models.classifier import DefaultRiskClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit resiliency — synthetic data and default-risk training"
    )
    parser.add_argument("--n-samples", type=int, default=10_000)
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

    logger.info("=== Step 1: Generating synthetic dataset ===")
    gen = CustomerDataGenerator(GeneratorConfig(n_samples=args.n_samples))
    df = gen.generate()
    train_df, test_df = gen.train_test_split(df)

    df.to_parquet(out_dir / "customers.parquet", index=False)
    logger.info("Dataset saved -> {}", out_dir / "customers.parquet")

    X_train = train_df.drop(columns=[LABEL_COL, "hardship_severity"])
    y_train = train_df[LABEL_COL]
    X_test = test_df.drop(columns=[LABEL_COL, "hardship_severity"])
    y_test = test_df[LABEL_COL]

    logger.info("=== Step 2: Training XGBoost default-risk classifier ===")
    clf = DefaultRiskClassifier(calibrate=True)
    clf.fit(X_train, y_train, eval_set=(X_test, y_test))

    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)

    logger.info("\n{}", business_metrics(y_test.values, y_prob, y_pred).T.to_string())
    logger.info("\n{}", classification_report_df(y_test.values, y_pred).to_string())

    if args.save_plots:
        plot_roc_curve(y_test.values, y_prob).savefig(
            plots_dir / "roc_curve.png", dpi=150
        )
        plot_confusion_matrix(y_test.values, y_pred).savefig(
            plots_dir / "confusion_matrix.png", dpi=150
        )
        plot_feature_importance(
            clf.feature_names_, clf.feature_importances_
        ).savefig(plots_dir / "feature_importance.png", dpi=150)
        logger.info("Plots saved -> {}", plots_dir)

    clf.save(out_dir / "default_risk_classifier.pkl")
    logger.success(
        "Training complete. Dataset and default-risk model saved to: {}", out_dir
    )


if __name__ == "__main__":
    main()
