"""
Data Loader Module
Handles downloading, caching, and loading of asset price data.
"""

import pandas as pd
from typing import List, Optional


def download_price_data(
    tickers: List[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Download historical daily price data (Adjusted Close) from Yahoo Finance.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols to download.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted close prices with tickers as columns and dates as index.
    """
    pass


def save_to_cache(data: pd.DataFrame, filepath: str) -> None:
    """
    Cache the downloaded price data to a local file.

    Parameters
    ----------
    data : pd.DataFrame
        Price DataFrame to cache.
    filepath : str
        Path to save the cached file (e.g. CSV or parquet format).
    """
    pass


def load_from_cache(filepath: str) -> pd.DataFrame:
    """
    Load price data from a cached local file.

    Parameters
    ----------
    filepath : str
        Path to the cached file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the cached price data.
    """
    pass
