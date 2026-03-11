"""
resiliency — Capital One Resiliency Intelligence library.

Modules
-------
data        : Synthetic customer data generation
models      : XGBoost default classifier and Q-learning RL agent
evaluation  : Metrics, ROC curves, confusion matrices
utils       : Preprocessing and feature engineering helpers
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("resiliency")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = ["data", "models", "evaluation", "utils"]
