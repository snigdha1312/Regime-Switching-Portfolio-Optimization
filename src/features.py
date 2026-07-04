"""
Features Module
Implements feature engineering for regime-switching models.
Calculates momentum and volatility features for NIFTY 50 and India VIX.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional


def build_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature set for the HMM from aligned prices.
    
    Expected columns in prices_df:
        - '^NSEI': NIFTY 50 price
        - '^INDIAVIX': India VIX price
        
    Parameters
    ----------
    prices_df : pd.DataFrame
        Aligned price DataFrame containing NIFTY and VIX.
        
    Returns
    -------
    pd.DataFrame
        DataFrame of features, with initial NaN rows dropped.
    """
    if "^NSEI" not in prices_df.columns:
        raise ValueError("NIFTY 50 index (^NSEI) must be in prices_df.")
    if "^INDIAVIX" not in prices_df.columns:
        raise ValueError("India VIX (^INDIAVIX) must be in prices_df.")

    nifty = prices_df["^NSEI"]
    vix = prices_df["^INDIAVIX"]
    
    # Calculate returns for volatility and momentum
    nifty_returns = nifty.pct_change()
    nifty_log_returns = np.log(nifty).diff()

    features = pd.DataFrame(index=prices_df.index)

    # 1. Momentum features:
    # 5-day, 21-day, and 63-day rolling cumulative returns
    # We use rolling sum of log returns and exponentiate to get simple cumulative returns
    features["nifty_momentum_5"] = np.exp(nifty_log_returns.rolling(5).sum()) - 1
    features["nifty_momentum_21"] = np.exp(nifty_log_returns.rolling(21).sum()) - 1
    features["nifty_momentum_63"] = np.exp(nifty_log_returns.rolling(63).sum()) - 1

    # Distance of current price from its 63-day and 200-day moving average (as %)
    nifty_ma_63 = nifty.rolling(63).mean()
    nifty_ma_200 = nifty.rolling(200).mean()
    
    features["nifty_ma_dist_63"] = ((nifty / nifty_ma_63) - 1) * 100
    features["nifty_ma_dist_200"] = ((nifty / nifty_ma_200) - 1) * 100

    # 2. Volatility features:
    # 21-day rolling standard deviation of daily returns (annualized)
    features["nifty_volatility_21"] = nifty_returns.rolling(21).std() * np.sqrt(252)

    # India VIX level itself and its 21-day rate of change
    features["vix_level"] = vix
    features["vix_roc_21"] = (vix / vix.shift(21)) - 1

    # Drop the initial NaN rows from the longest rolling window
    raw_len = len(features)
    features_clean = features.dropna()
    dropped_rows = raw_len - len(features_clean)
    
    print("\n--- Feature Engineering Statistics ---")
    print(f"Total raw features rows: {raw_len}")
    print(f"Cleaned features rows: {len(features_clean)}")
    print(f"Dropped initial NaN rows: {dropped_rows}")
    print("--------------------------------------\n")

    return features_clean


def standardize_features(
    features_df: pd.DataFrame, fit_start: str, fit_end: str
) -> pd.DataFrame:
    """
    Standardize features (z-score) by fitting the mean and standard deviation
    only on the specified training date range, and then scaling the entire DataFrame.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame of features.
    fit_start : str
        Start date of the fitting window ('YYYY-MM-DD').
    fit_end : str
        End date of the fitting window ('YYYY-MM-DD').

    Returns
    -------
    pd.DataFrame
        Standardized features DataFrame.
    """
    # Verify dates are in index
    fit_data = features_df.loc[fit_start:fit_end]
    if fit_data.empty:
        raise ValueError(
            f"No data available in fit range {fit_start} to {fit_end} for scaling."
        )
        
    mean = fit_data.mean()
    std = fit_data.std()
    
    # Avoid division by zero for flat features (if any)
    std = std.replace(0, 1e-8)
    
    standardized_df = (features_df - mean) / std
    return standardized_df
