"""
Standalone data generation script.

Usage
-----
    python scripts/generate_data.py [--n-samples 10000] [--output data/customers.parquet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from resiliency.data.generator import CustomerDataGenerator, GeneratorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic credit hardship customer data")
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--default-rate", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/customers.parquet")
    args = parser.parse_args()

    cfg = GeneratorConfig(
        n_samples=args.n_samples,
        default_rate=args.default_rate,
        random_seed=args.seed,
    )
    gen = CustomerDataGenerator(cfg)
    df = gen.generate()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Dataset saved → {out}  (shape={df.shape})")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
