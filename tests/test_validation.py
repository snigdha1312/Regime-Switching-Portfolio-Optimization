"""
Unit tests for the validation module.
Verifies that the walk-forward validation harness and leakage check work correctly.
"""

import numpy as np
import pandas as pd
from src.validation import run_walk_forward_validation, run_leakage_check


def test_walk_forward_no_leakage():
    """
    Test that the walk-forward validation correctly partitions training and testing,
    and passes the automated leakage check.
    """
    # Create 900 days of dummy feature data
    # (Must be > 756 days initial training window)
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=900, freq="D")
    
    # 8 features matching our structure
    features_data = np.random.normal(0, 1, size=(900, 8))
    cols = [
        "nifty_momentum_5", "nifty_momentum_21", "nifty_momentum_63",
        "nifty_ma_dist_63", "nifty_ma_dist_200",
        "nifty_volatility_21", "vix_level", "vix_roc_21"
    ]
    dummy_features = pd.DataFrame(features_data, index=dates, columns=cols)
    
    # Run walk-forward validation
    # Train on 756 days, predict next 63 days
    oos_regimes, window_records = run_walk_forward_validation(
        dummy_features,
        initial_train_days=756,
        refit_freq=63,
        random_state=100
    )
    
    # Expected OOS length: 900 - 756 = 144 days
    assert len(oos_regimes) == 144
    assert oos_regimes.index[0] == dates[756]
    assert oos_regimes.index[-1] == dates[899]
    
    # Run leakage checks
    success = run_leakage_check(oos_regimes, window_records, n_samples=10)
    assert success is True, "Walk-forward validation failed the leakage check!"
