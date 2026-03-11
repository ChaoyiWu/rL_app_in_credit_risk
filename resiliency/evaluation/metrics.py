"""
Evaluation metrics and visualisations for the default risk classifier.

All plot functions return a ``matplotlib.figure.Figure`` so they can be
rendered inline in notebooks or saved to disk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    average_precision_score,
    brier_score_loss,
)
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
_PALETTE = {"primary": "#004977", "secondary": "#D03027", "accent": "#F5A623"}
sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "XGBoost",
    figsize: tuple[int, int] = (7, 6),
) -> plt.Figure:
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    model_name : str
        Label shown in the legend.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        fpr, tpr,
        color=_PALETTE["primary"],
        lw=2,
        label=f"{model_name}  (AUC = {roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=_PALETTE["primary"])

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.06])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Default Risk Classifier", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    figsize: tuple[int, int] = (6, 5),
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot an annotated confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list[str], optional
        Class names. Defaults to ["No Default", "Default"].
    figsize : tuple
        Figure size.
    normalize : bool
        Show row-normalised rates instead of raw counts.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = labels or ["No Default", "Default"]
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".1%"
        vmax = 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmax = cm.max()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    title = "Confusion Matrix" + (" (Row-Normalised)" if normalize else "")
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Classification report as DataFrame
# ---------------------------------------------------------------------------

def classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Return sklearn's classification report as a tidy DataFrame.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    target_names : list[str], optional
        Class display names.

    Returns
    -------
    pd.DataFrame
    """
    target_names = target_names or ["No Default", "Default"]
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    return pd.DataFrame(report).T.round(3)


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    feature_names: Sequence[str],
    importances: np.ndarray,
    top_n: int = 20,
    figsize: tuple[int, int] = (9, 7),
) -> plt.Figure:
    """
    Horizontal bar chart of feature importances.

    Parameters
    ----------
    feature_names : list[str]
        Names aligned with ``importances``.
    importances : np.ndarray
        Importance scores (e.g., XGBoost gain).
    top_n : int
        Number of top features to display.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        range(len(top_names)),
        top_vals[::-1],
        color=_PALETTE["primary"],
        edgecolor="white",
        height=0.7,
    )
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title(
        f"Top {top_n} Feature Importances — Default Risk Model",
        fontsize=13,
        fontweight="bold",
    )
    # Annotate bars
    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Business-oriented summary metrics
# ---------------------------------------------------------------------------

def business_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    avg_balance: float = 4_500.0,
    collection_cost_rate: float = 0.25,
) -> pd.DataFrame:
    """
    Compute business-relevant metrics for the credit resiliency programme.

    Parameters
    ----------
    y_true : array-like
        True default labels.
    y_prob : array-like
        Predicted default probabilities.
    y_pred : array-like
        Binary predictions at chosen threshold.
    avg_balance : float
        Average account balance (USD) — used for loss estimation.
    collection_cost_rate : float
        Cost of collections as fraction of balance.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with key business metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    roc_auc = auc(*roc_curve(y_true, y_prob)[:2])
    avg_precision = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # Estimated dollar impact
    missed_defaults = fn  # defaults not caught → full loss
    false_alerts = fp     # good customers incorrectly flagged → friction cost
    estimated_loss_reduction = tp * avg_balance * (1 - collection_cost_rate)
    intervention_cost = (tp + fp) * avg_balance * 0.02  # 2% programme cost

    return pd.DataFrame(
        {
            "ROC-AUC": [round(roc_auc, 4)],
            "Avg Precision (PR-AUC)": [round(avg_precision, 4)],
            "Brier Score": [round(brier, 4)],
            "Precision": [round(precision, 4)],
            "Recall (Sensitivity)": [round(recall, 4)],
            "F1 Score": [round(f1, 4)],
            "True Positives": [int(tp)],
            "False Negatives (Missed Defaults)": [int(fn)],
            "False Positives (False Alerts)": [int(fp)],
            "Est. Loss Reduction ($)": [f"${estimated_loss_reduction:,.0f}"],
            "Est. Intervention Cost ($)": [f"${intervention_cost:,.0f}"],
        }
    )
