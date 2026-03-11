"""Setup configuration for the resiliency library."""
from setuptools import setup, find_packages

setup(
    name="resiliency",
    version="0.1.0",
    description="Resiliency Intelligence — credit risk and debt resolution library",
    packages=find_packages(include=["resiliency", "resiliency.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.11.0",
        "gymnasium>=0.29.0",
        "joblib>=1.3.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "api": ["fastapi>=0.104.0", "uvicorn[standard]>=0.24.0", "pydantic>=2.4.0"],
        "rl": ["stable-baselines3>=2.1.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "httpx>=0.25.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
