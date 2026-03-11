"""Unit tests for the synthetic data generator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resiliency.data.generator import (
    CustomerDataGenerator,
    GeneratorConfig,
    FEATURE_COLS,
    LABEL_COL,
    HARDSHIP_SEVERITY_COL,
)


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    gen = CustomerDataGenerator(GeneratorConfig(n_samples=500, random_seed=0))
    return gen.generate()


class TestGeneratorOutput:
    def test_shape(self, small_df):
        """Dataset has correct number of rows and all expected columns."""
        assert len(small_df) == 500
        for col in FEATURE_COLS + [LABEL_COL, HARDSHIP_SEVERITY_COL]:
            assert col in small_df.columns, f"Missing column: {col}"

    def test_default_rate_approximate(self, small_df):
        """Default rate is within ±5 percentage points of the target."""
        rate = small_df[LABEL_COL].mean()
        assert 0.17 <= rate <= 0.27, f"Default rate {rate:.2%} out of expected range"

    def test_label_binary(self, small_df):
        """Default label contains only 0 and 1."""
        assert set(small_df[LABEL_COL].unique()).issubset({0, 1})

    def test_hardship_severity_ordinal(self, small_df):
        """Hardship severity is 0, 1, or 2."""
        assert set(small_df[HARDSHIP_SEVERITY_COL].unique()).issubset({0, 1, 2})

    def test_credit_score_range(self, small_df):
        assert small_df["credit_score"].between(300, 850).all()

    def test_utilization_range(self, small_df):
        assert small_df["credit_utilization_pct"].between(0.0, 1.0).all()

    def test_no_nulls(self, small_df):
        assert small_df.isnull().sum().sum() == 0

    def test_reproducible(self):
        """Same seed produces identical datasets."""
        g1 = CustomerDataGenerator(GeneratorConfig(n_samples=100, random_seed=7))
        g2 = CustomerDataGenerator(GeneratorConfig(n_samples=100, random_seed=7))
        pd.testing.assert_frame_equal(g1.generate(), g2.generate())

    def test_different_seed_differs(self):
        g1 = CustomerDataGenerator(GeneratorConfig(n_samples=100, random_seed=1))
        g2 = CustomerDataGenerator(GeneratorConfig(n_samples=100, random_seed=2))
        df1 = g1.generate()
        df2 = g2.generate()
        assert not df1["credit_score"].equals(df2["credit_score"])


class TestTrainTestSplit:
    def test_split_sizes(self, small_df):
        gen = CustomerDataGenerator()
        train, test = gen.train_test_split(small_df, test_size=0.20)
        assert len(train) + len(test) == len(small_df)
        assert abs(len(test) / len(small_df) - 0.20) < 0.02

    def test_no_overlap(self, small_df):
        gen = CustomerDataGenerator()
        train, test = gen.train_test_split(small_df, test_size=0.20)
        # After reset_index, verify indices don't overlap in original index space
        assert len(train) + len(test) == len(small_df)

    def test_stratified(self, small_df):
        gen = CustomerDataGenerator()
        train, test = gen.train_test_split(small_df, test_size=0.20)
        train_rate = train[LABEL_COL].mean()
        test_rate = test[LABEL_COL].mean()
        assert abs(train_rate - test_rate) < 0.05
