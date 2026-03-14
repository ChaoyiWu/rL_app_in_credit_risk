"""Model evaluation — ROC curves, confusion matrices, business metrics, and OPE."""
from .metrics import (
    plot_roc_curve,
    plot_confusion_matrix,
    classification_report_df,
    plot_feature_importance,
    business_metrics,
)
from .ips import (
    importance_weights,
    ips_estimate,
    snips_estimate,
    clipped_ips_estimate,
    effective_sample_size,
)
from .ope import OPEEvaluator

# Convenience alias — compute_ips_weights is the public-facing name;
# importance_weights is the canonical implementation in ips.py.
compute_ips_weights = importance_weights

__all__ = [
    "plot_roc_curve",
    "plot_confusion_matrix",
    "classification_report_df",
    "plot_feature_importance",
    "business_metrics",
    # IPS estimators
    "importance_weights",
    "compute_ips_weights",
    "ips_estimate",
    "snips_estimate",
    "clipped_ips_estimate",
    "effective_sample_size",
    # OPE
    "OPEEvaluator",
]
