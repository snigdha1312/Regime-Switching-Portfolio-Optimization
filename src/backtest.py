"""
Backtest Module
Implements backtesting mechanics, including weight rebalancing, asset price drift,
transaction costs, and benchmark simulations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional


def run_dynamic_backtest(
    prices_df: pd.DataFrame,
    oos_regimes: pd.Series,
    target_weights_df: pd.DataFrame,
    transaction_cost_pct: float = 0.00075,  # 7.5 bps
    min_hold_days: int = 5,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Simulate a portfolio backtest with dynamic regime-based rebalancing,
    asset weight drift, and transaction costs.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Aligned price DataFrame for the tradable assets.
    oos_regimes : pd.Series
        Out-of-sample regime series.
    target_weights_df : pd.DataFrame
        DataFrame of target weights for each date.
    transaction_cost_pct : float, default 0.00075
        Transaction cost as a fraction of notional traded (e.g. 7.5 bps = 0.00075).
    min_hold_days : int, default 5
        Minimum number of trading days to hold a portfolio before rebalancing.

    Returns
    -------
    equity_with_costs : pd.Series
        Portfolio value series over time with transaction costs.
    equity_no_costs : pd.Series
        Portfolio value series over time without transaction costs.
    daily_turnover : pd.Series
        Series of daily turnover values.
    """
    dates = oos_regimes.index
    n_days = len(dates)
    
    # Calculate daily asset simple returns
    returns_df = prices_df[target_weights_df.columns].pct_change().loc[dates].fillna(0.0)
    
    # Initialize variables for portfolio simulation
    val_c = 1.0  # value with costs
    val_nc = 1.0  # value without costs
    
    # Initial target weights
    w_c = target_weights_df.iloc[0].values.copy()
    w_nc = target_weights_df.iloc[0].values.copy()
    
    # Apply initial portfolio build cost (transition from 100% cash to target weights)
    initial_turnover = np.sum(w_c)
    initial_cost = initial_turnover * transaction_cost_pct
    val_c = val_c - initial_cost
    
    history_c = [val_c]
    history_nc = [val_nc]
    turnovers = [initial_turnover]
    
    last_rebalance_idx = 0
    last_rebalance_regime = oos_regimes.iloc[0]
    
    # Drifted weights trackers
    w_c_drift = w_c.copy()
    w_nc_drift = w_nc.copy()
    
    for t in range(1, n_days):
        date = dates[t]
        asset_rets = returns_df.iloc[t].values
        
        # 1. Earn returns on previous day's weights
        p_ret_c = np.sum(w_c * asset_rets)
        p_ret_nc = np.sum(w_nc * asset_rets)
        
        val_c = val_c * (1.0 + p_ret_c)
        val_nc = val_nc * (1.0 + p_ret_nc)
        
        # 2. Drift the weights as prices move
        w_c_drift = w_c * (1.0 + asset_rets)
        w_c_drift = w_c_drift / np.sum(w_c_drift) if np.sum(w_c_drift) > 0 else w_c
        
        w_nc_drift = w_nc * (1.0 + asset_rets)
        w_nc_drift = w_nc_drift / np.sum(w_nc_drift) if np.sum(w_nc_drift) > 0 else w_nc
        
        # 3. Check if we should rebalance (using day t's predicted regime and target weights)
        current_regime = oos_regimes.iloc[t]
        days_since_rebalance = t - last_rebalance_idx
        
        rebalanced_today = False
        turnover_t = 0.0
        
        # Rebalance conditions:
        # - Regime changed AND minimum holding period has elapsed
        if (current_regime != last_rebalance_regime) and (days_since_rebalance >= min_hold_days):
            target_w = target_weights_df.iloc[t].values
            
            # Compute turnover: sum(|target - drifted|)
            turnover_t = np.sum(np.abs(target_w - w_c_drift))
            cost_t = val_c * turnover_t * transaction_cost_pct
            
            # Deduct costs and update weights
            val_c = val_c - cost_t
            w_c = target_w.copy()
            w_nc = target_w.copy()  # Update target weights for no-cost portfolio too
            
            last_rebalance_idx = t
            last_rebalance_regime = current_regime
            rebalanced_today = True
        else:
            # No rebalance, weights continue to drift
            w_c = w_c_drift.copy()
            w_nc = w_nc_drift.copy()
            
        history_c.append(val_c)
        history_nc.append(val_nc)
        turnovers.append(turnover_t if rebalanced_today else 0.0)

    equity_c = pd.Series(history_c, index=dates, name="strategy_with_costs")
    equity_nc = pd.Series(history_nc, index=dates, name="strategy_no_costs")
    turnover_series = pd.Series(turnovers, index=dates, name="turnover")
    
    return equity_c, equity_nc, turnover_series


def run_static_backtest(
    prices_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    target_weights: np.ndarray,
    rebalance_freq: int = 21,  # monthly
    transaction_cost_pct: float = 0.00075,
) -> Tuple[pd.Series, pd.Series]:
    """
    Simulate a static benchmark portfolio rebalanced on a fixed calendar schedule.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Aligned price DataFrame for the assets.
    dates : pd.DatetimeIndex
        Out-of-sample dates to run the backtest.
    target_weights : np.ndarray
        Fixed allocation weights (e.g. [0.6, 0.0, 0.4] for 60/40).
    rebalance_freq : int, default 21
        Rebalancing frequency in trading days.
    transaction_cost_pct : float, default 0.00075
        Transaction cost percentage.

    Returns
    -------
    equity : pd.Series
        Benchmark equity curve.
    turnover_series : pd.Series
        Series of benchmark daily turnover.
    """
    n_days = len(dates)
    returns_df = prices_df.pct_change().loc[dates].fillna(0.0).values
    
    val = 1.0
    w = np.array(target_weights, dtype=float)
    
    # Apply initial portfolio build cost
    initial_turnover = np.sum(w)
    initial_cost = initial_turnover * transaction_cost_pct
    val = val - initial_cost
    
    history = [val]
    turnovers = [initial_turnover]
    
    w_drift = w.copy()
    last_rebalance_idx = 0
    
    for t in range(1, n_days):
        asset_rets = returns_df[t]
        
        # Earn returns
        p_ret = np.sum(w * asset_rets)
        val = val * (1.0 + p_ret)
        
        # Drift weights
        w_drift = w * (1.0 + asset_rets)
        w_drift = w_drift / np.sum(w_drift) if np.sum(w_drift) > 0 else w
        
        # Rebalance check
        days_since_rebalance = t - last_rebalance_idx
        if days_since_rebalance >= rebalance_freq:
            turnover_t = np.sum(np.abs(target_weights - w_drift))
            cost_t = val * turnover_t * transaction_cost_pct
            
            val = val - cost_t
            w = np.array(target_weights, dtype=float)
            last_rebalance_idx = t
            rebalanced = True
        else:
            w = w_drift.copy()
            rebalanced = False
            turnover_t = 0.0
            
        history.append(val)
        turnovers.append(turnover_t if rebalanced else 0.0)
        
    equity = pd.Series(history, index=dates)
    turnover_series = pd.Series(turnovers, index=dates)
    return equity, turnover_series
