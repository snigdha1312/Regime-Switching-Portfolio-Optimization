"""
Unit tests for the backtest module.
Verifies portfolio drift, rebalancing limits, and transaction cost calculations.
"""

import numpy as np
import pandas as pd
import pytest
from src.backtest import run_dynamic_backtest, run_static_backtest


def test_backtest_simulation():
    """
    Verify that backtest simulations compute correctly, penalize transaction costs,
    and respect minimum hold limits.
    """
    # 50 days of dummy price data for 3 assets
    dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
    
    # Simulating constant asset returns to trace calculations exactly
    # Let's say returns are 0.1% daily for asset 0, 0.2% for asset 1, 0.3% for asset 2
    prices_raw = []
    p0, p1, p2 = 100.0, 100.0, 100.0
    for t in range(50):
        prices_raw.append([p0, p1, p2])
        p0 *= 1.001
        p1 *= 1.002
        p2 *= 1.003
        
    prices_df = pd.DataFrame(prices_raw, index=dates, columns=["Asset_A", "Asset_B", "Asset_C"])
    
    # Target weights DataFrame
    target_weights = pd.DataFrame(
        [[0.5, 0.3, 0.2]] * 50,
        index=dates,
        columns=["Asset_A", "Asset_B", "Asset_C"]
    )
    
    # Alternate regime every 3 days to test min_hold limit of 5 days
    # Regime flips: Day 0 (Bull), Day 3 (Bear), Day 6 (Bull), Day 9 (Bear)
    regimes = []
    current_reg = "Bull"
    for i in range(50):
        if i > 0 and i % 3 == 0:
            current_reg = "Bear" if current_reg == "Bull" else "Bull"
        regimes.append(current_reg)
    oos_regimes = pd.Series(regimes, index=dates)
    
    # Run backtest with 1% transaction cost to make it highly visible
    equity_c, equity_nc, turnover = run_dynamic_backtest(
        prices_df=prices_df,
        oos_regimes=oos_regimes,
        target_weights_df=target_weights,
        transaction_cost_pct=0.01,
        min_hold_days=5
    )
    
    # Verify outputs
    assert len(equity_c) == 50
    assert len(equity_nc) == 50
    assert len(turnover) == 50
    
    # First day has initial build turnover of 1.0, cost = 0.01. So equity_c starts at 0.99
    assert equity_c.iloc[0] == pytest.approx(0.99)
    assert equity_nc.iloc[0] == pytest.approx(1.00)
    
    # Over time, equity with costs should be strictly less than equity without costs
    assert equity_c.iloc[-1] < equity_nc.iloc[-1]
    
    # Check that rebalancing did not occur at Day 3 (only 3 days elapsed, min_hold is 5)
    # Day 5: 5 days have elapsed, so it rebalances to Bear.
    # Day 6 goes back to "Bull" but we cannot rebalance (only 1 day since last rebalance).
    assert turnover.iloc[3] == 0.0
    assert turnover.iloc[5] > 0.0
    assert turnover.iloc[6] == 0.0
    
    # Run static backtest (60/40 target)
    bench_eq, bench_to = run_static_backtest(
        prices_df=prices_df,
        dates=dates,
        target_weights=np.array([0.6, 0.0, 0.4]),
        rebalance_freq=10,
        transaction_cost_pct=0.01
    )
    
    assert len(bench_eq) == 50
    assert bench_eq.iloc[0] == pytest.approx(0.99)  # 60 + 40 = 100% build cost
    
    print("Backtest unit tests passed successfully.")
