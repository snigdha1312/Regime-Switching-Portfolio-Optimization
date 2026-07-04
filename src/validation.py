"""
Validation Module
Computes performance metrics, risk statistics, and validates the regime-switching pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def calculate_performance_metrics(
    portfolio_value: pd.Series,
    benchmark_value: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics: Sharpe ratio, Sortino ratio, Max Drawdown,
    Calmar ratio, Annualized Return, and Annualized Volatility.

    Parameters
    ----------
    portfolio_value : pd.Series
        Daily portfolio equity curve (values or cumulative returns).
    benchmark_value : pd.Series, optional
        Daily benchmark equity curve.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate of return.

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed performance and risk statistics.
    """
    pass


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate the maximum peak-to-trough drawdown of an equity curve.

    Parameters
    ----------
    equity_curve : pd.Series
        Daily portfolio values.

    Returns
    -------
    float
        Maximum drawdown as a positive fraction (e.g. 0.15 for 15%).
    """
    pass


def walk_forward_split(
    data: pd.DataFrame, n_splits: int = 5, train_size_pct: float = 0.6
) -> list:
    """
    Generate train/test indices for walk-forward time-series cross-validation.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to split.
    n_splits : int, default 5
        Number of walk-forward validation folds.
    train_size_pct : float, default 0.6
        Percentage of historical data used for initial training.

    Returns
    -------
    list of tuple
        List containing (train_indices, test_indices) for each fold.
    """
    pass
