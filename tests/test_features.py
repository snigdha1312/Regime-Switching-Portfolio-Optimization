"""
Unit tests for the features module.
Ensures that no feature calculation has look-ahead bias.
"""

import numpy as np
import pandas as pd
import pytest
from src.features import build_features


def test_no_lookahead_bias():
    """
    Verify that features at time t only depend on historical data up to time t
    and are not affected by changes to future prices/returns.
    """
    # Create 500 days of dummy price data
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
    
    # Simulate a random walk for equity and mean-reverting VIX
    nifty_prices = 10000 + np.cumsum(np.random.normal(5, 50, size=500))
    vix_prices = 15 + np.random.normal(0, 2, size=500)
    
    dummy_prices = pd.DataFrame(
        {"^NSEI": nifty_prices, "^INDIAVIX": vix_prices}, index=dates
    )
    
    # Calculate baseline features
    baseline_features = build_features(dummy_prices)
    
    # Choose test point t and future point t_future
    # Must be within the valid range of baseline_features (size is 301 after 199 drop)
    t_idx = 100
    t_date = baseline_features.index[t_idx]
    
    t_future_idx = 250
    t_future_date = baseline_features.index[t_future_idx]
    
    # Assert future index is indeed greater than t index
    assert t_future_date > t_date
    
    # Perturb the future price of NIFTY and VIX
    perturbed_prices = dummy_prices.copy()
    perturbed_prices.loc[t_future_date, "^NSEI"] += 10000.0
    perturbed_prices.loc[t_future_date, "^INDIAVIX"] += 50.0
    
    # Calculate features on perturbed data
    perturbed_features = build_features(perturbed_prices)
    
    # Check that for all features, the value at time t is unchanged
    for col in baseline_features.columns:
        baseline_val = baseline_features.loc[t_date, col]
        perturbed_val = perturbed_features.loc[t_date, col]
        
        # Using pytest's approx for floating point precision
        assert baseline_val == pytest.approx(perturbed_val), (
            f"Look-ahead bias detected in feature '{col}'! "
            f"Baseline: {baseline_val}, Perturbed: {perturbed_val} at date {t_date}"
        )
        
    print(f"Verified no look-ahead bias for all features at date {t_date}.")
