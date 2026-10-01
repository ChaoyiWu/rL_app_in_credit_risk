"""Customer context features used by the treatment-selection model."""
from __future__ import annotations

import numpy as np
import pandas as pd


CUSTOMER_CONTEXT_FEATURES = [
    "credit_score",
    "credit_utilization_pct",
    "months_delinquent",
    "debt_to_income_ratio",
    "num_missed_payments_12m",
    "has_bankruptcy",
    "requested_hardship_program",
    "hardship_severity",
    "annual_income",
    "min_payment_ratio",
]

_CONTEXT_SCALE = np.array(
    [850, 1.0, 24, 5.0, 12, 1, 1, 2, 300_000, 3.0],
    dtype=np.float32,
)


def extract_customer_context(customer: dict | pd.Series) -> np.ndarray:
    """Return a normalized 10-feature context vector for one customer."""
    raw = np.array(
        [float(customer.get(f, 0)) for f in CUSTOMER_CONTEXT_FEATURES],
        dtype=np.float32,
    )
    return np.clip(raw / (_CONTEXT_SCALE + 1e-9), 0.0, 1.0)
