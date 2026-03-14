"""
Importance-weighted estimators for off-policy evaluation.

Background
----------
When a logging policy π_0 collected historical data but we want to estimate
the value of a *target* policy π_1, we cannot average rewards directly —
π_0 may have chosen different actions than π_1 would.  Importance Sampling
(IS) corrects for this distributional shift by re-weighting each observed
reward by how much more (or less) likely π_1 is to have chosen that action:

    w_i  =  π_1(a_i | x_i) / π_0(a_i | x_i)

All estimators in this module accept **per-action propensities** — the
scalar probability of the *actually chosen* action under each policy —
rather than full action-probability distributions.  If you have a full
probability matrix of shape (n, n_arms), slice the relevant column before
calling these functions:

    π_0_chosen = propensity_matrix[np.arange(n), actions]

Estimators
----------
ips_estimate
    Standard IPS: V̂_IPS = (1/n) Σ w_i r_i
    Unbiased but can have high variance when weights are large.

snips_estimate  (Self-Normalised IPS)
    V̂_SNIPS = Σ(w_i r_i) / Σ w_i
    Biased but often lower variance than plain IPS; bounded in [r_min, r_max].

clipped_ips_estimate
    V̂_cIPS: same as IPS but weights are capped at ``clip`` to reduce
    extreme-weight variance at the cost of some bias.

importance_weights
    Helper that returns the raw weight vector (optionally clipped).
"""
from __future__ import annotations

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Core weight helper
# ---------------------------------------------------------------------------

def importance_weights(
    historical_propensity: np.ndarray,
    new_propensity: np.ndarray,
    clip: float | None = None,
) -> np.ndarray:
    """
    Compute per-sample importance weights  w_i = π_1(a_i) / π_0(a_i).

    Parameters
    ----------
    historical_propensity : np.ndarray of shape (n,)
        Probability of the *chosen* action under the logging policy π_0.
        Must be strictly positive (raises if any value ≤ 0).
    new_propensity : np.ndarray of shape (n,)
        Probability of the same action under the target policy π_1.
    clip : float, optional
        If set, weights are capped at this value to reduce variance.
        A common choice is ``clip=10.0``.

    Returns
    -------
    np.ndarray of shape (n,)
        Importance weights, optionally clipped.

    Raises
    ------
    ValueError
        If any ``historical_propensity`` value is ≤ 0.
    """
    historical_propensity = np.asarray(historical_propensity, dtype=np.float64)
    new_propensity = np.asarray(new_propensity, dtype=np.float64)

    if np.any(historical_propensity <= 0):
        raise ValueError(
            "historical_propensity contains non-positive values. "
            "Logging propensities must be > 0 to avoid division by zero."
        )

    weights = new_propensity / historical_propensity

    if clip is not None:
        if clip <= 0:
            raise ValueError(f"clip must be positive, got {clip}")
        n_clipped = int(np.sum(weights > clip))
        if n_clipped > 0:
            logger.debug(
                "importance_weights: clipped {} / {} weights at {:.1f}",
                n_clipped,
                len(weights),
                clip,
            )
        weights = np.minimum(weights, clip)

    return weights


# ---------------------------------------------------------------------------
# IPS estimators
# ---------------------------------------------------------------------------

def ips_estimate(
    rewards: np.ndarray,
    historical_propensity: np.ndarray,
    new_propensity: np.ndarray,
    clip: float | None = None,
) -> float:
    """
    Standard Importance-Weighted Policy Value estimator.

    .. math::

        \\hat{V}_{\\text{IPS}} = \\frac{1}{n} \\sum_{i=1}^{n}
            \\frac{\\pi_1(a_i \\mid x_i)}{\\pi_0(a_i \\mid x_i)} r_i

    Unbiased under the assumption that  π_0(a|x) > 0  whenever  π_1(a|x) > 0
    (the *coverage* condition).

    Parameters
    ----------
    rewards : np.ndarray of shape (n,)
        Observed scalar rewards from the logged interactions.
    historical_propensity : np.ndarray of shape (n,)
        Probability of the chosen action under the logging policy π_0.
    new_propensity : np.ndarray of shape (n,)
        Probability of the same action under the target policy π_1.
    clip : float, optional
        Weight clipping threshold.  See :func:`importance_weights`.

    Returns
    -------
    float
        Estimated policy value.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    w = importance_weights(historical_propensity, new_propensity, clip=clip)
    estimate = float(np.mean(w * rewards))
    logger.debug(
        "IPS estimate={:.4f}  |  w_mean={:.3f}  w_max={:.3f}",
        estimate, w.mean(), w.max(),
    )
    return estimate


def snips_estimate(
    rewards: np.ndarray,
    historical_propensity: np.ndarray,
    new_propensity: np.ndarray,
    clip: float | None = None,
) -> float:
    """
    Self-Normalised IPS (SNIPS) estimator.

    .. math::

        \\hat{V}_{\\text{SNIPS}} =
            \\frac{\\sum_i w_i r_i}{\\sum_i w_i}

    Biased but typically lower variance than plain IPS.  The normaliser
    Σ w_i / n estimates the expected weight under π_0, and dividing by it
    removes one source of scale error.

    Parameters
    ----------
    rewards : np.ndarray of shape (n,)
    historical_propensity : np.ndarray of shape (n,)
    new_propensity : np.ndarray of shape (n,)
    clip : float, optional

    Returns
    -------
    float
        Estimated policy value.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    w = importance_weights(historical_propensity, new_propensity, clip=clip)
    w_sum = w.sum()
    if w_sum == 0:
        logger.warning("snips_estimate: all importance weights are zero — returning 0.0")
        return 0.0
    estimate = float((w * rewards).sum() / w_sum)
    logger.debug("SNIPS estimate={:.4f}", estimate)
    return estimate


def clipped_ips_estimate(
    rewards: np.ndarray,
    historical_propensity: np.ndarray,
    new_propensity: np.ndarray,
    clip: float = 10.0,
) -> float:
    """
    IPS with hard weight clipping for variance reduction.

    Convenience wrapper around :func:`ips_estimate` with ``clip`` required
    (defaults to 10.0, a common choice in the causal-inference literature).

    Parameters
    ----------
    rewards : np.ndarray of shape (n,)
    historical_propensity : np.ndarray of shape (n,)
    new_propensity : np.ndarray of shape (n,)
    clip : float
        Maximum allowed weight.

    Returns
    -------
    float
        Clipped IPS estimate.
    """
    return ips_estimate(rewards, historical_propensity, new_propensity, clip=clip)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def effective_sample_size(
    historical_propensity: np.ndarray,
    new_propensity: np.ndarray,
    clip: float | None = None,
) -> float:
    """
    Effective Sample Size (ESS) — measures how many *effective* observations
    the re-weighted sample corresponds to.

    .. math::

        \\text{ESS} = \\frac{(\\sum_i w_i)^2}{\\sum_i w_i^2}

    A low ESS (close to 1) indicates that a few samples dominate the estimate
    and the IPS variance will be high.

    Parameters
    ----------
    historical_propensity : np.ndarray of shape (n,)
    new_propensity : np.ndarray of shape (n,)
    clip : float, optional

    Returns
    -------
    float
        ESS in [1, n].
    """
    w = importance_weights(historical_propensity, new_propensity, clip=clip)
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return ess
