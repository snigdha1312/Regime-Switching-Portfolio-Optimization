"""
Features Module
Implements feature engineering for regime-switching models.
"""

import pandas as pd
import numpy as np
from typing import List, Union


def calculate_returns(
    prices: pd.DataFrame, method: str = "log"
) -> pd.DataFrame:
    """
    Calculate asset returns from price data.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices.
    method : str, default 'log'
        Type of returns to compute, either 'simple' or 'log'.

    Returns
    -------
    pd.DataFrame
        DataFrame of returns.
    """
    pass


def calculate_rolling_volatility(
    returns: pd.DataFrame, window: int = 20
) -> pd.DataFrame:
    """
    Calculate rolling historical volatility of asset returns.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns.
    window : int, default 20
        Lookback window size in trading days.

    Returns
    -------
    pd.DataFrame
        DataFrame of rolling volatility (annualized).
    """
    pass


def calculate_technical_indicators(
    prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate standard technical indicators (e.g., Moving Average Crossover, RSI, MACD)
    to serve as features for the regime detection model.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated technical features.
    """
    pass


def prepare_features(
    prices: pd.DataFrame, feature_list: List[str]
) -> pd.DataFrame:
    """
    Combine returns, volatility, and other indicators to prepare the feature matrix
    for regime modeling.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices.
    feature_list : List[str]
        List of feature names to include (e.g., ['returns', 'volatility', 'rsi']).

    Returns
    -------
    pd.DataFrame
        Cleaned feature matrix with aligned dates and no missing values.
    """
    pass
