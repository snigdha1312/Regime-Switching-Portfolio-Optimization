"""
Data Loader Module
Handles downloading, caching, and loading of asset price data from yfinance,
aligning datasets, forward-filling gaps under tight constraints, and calculating log returns.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Optional


def clean_ticker_name(ticker: str) -> str:
    """Clean ticker name to be safe for filenames."""
    return ticker.replace("^", "").replace(".", "_")


def download_price_data(
    ticker: str, start_date: str, end_date: str, cache_dir: str = "data/raw"
) -> pd.DataFrame:
    """
    Download historical daily price data from Yahoo Finance and cache to parquet.

    Parameters
    ----------
    ticker : str
        Ticker symbol to download.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    cache_dir : str, default 'data/raw'
        Directory to cache raw downloads as parquet.

    Returns
    -------
    pd.DataFrame
        DataFrame of downloaded price data.
    """
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{clean_ticker_name(ticker)}.parquet"
    filepath = os.path.join(cache_dir, filename)

    # Load from cache if it exists and covers the requested range
    if os.path.exists(filepath):
        try:
            df = pd.read_parquet(filepath)
            if not df.empty:
                # Ensure the index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                cache_start = df.index.min().strftime("%Y-%m-%d")
                cache_end = df.index.max().strftime("%Y-%m-%d")
                
                if cache_start <= start_date and cache_end >= end_date:
                    print(f"Loaded '{ticker}' from cache ({cache_start} to {cache_end}).")
                    return df
        except Exception as e:
            print(f"Failed to read cache for '{ticker}': {e}. Downloading fresh data.")

    print(f"Downloading '{ticker}' from yfinance ({start_date} to {end_date})...")
    # Use auto_adjust=False to retrieve 'Adj Close'
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    
    if df.empty:
        raise ValueError(f"No data returned from yfinance for ticker '{ticker}'")
    
    # Save to parquet cache
    df.to_parquet(filepath)
    return df


def extract_price_series(df: pd.DataFrame) -> pd.Series:
    """
    Extract the Adjusted Close price series from a downloaded DataFrame,
    handling potential MultiIndex columns.
    """
    if isinstance(df.columns, pd.MultiIndex):
        metrics = df.columns.get_level_values(0)
        if "Adj Close" in metrics:
            series = df["Adj Close"]
        elif "Close" in metrics:
            series = df["Close"]
        else:
            series = df.iloc[:, 0]
        
        if isinstance(series, pd.DataFrame):
            series = series.squeeze()
    else:
        if "Adj Close" in df.columns:
            series = df["Adj Close"]
        elif "Close" in df.columns:
            series = df["Close"]
        else:
            series = df.iloc[:, 0]
            
    # Ensure index is datetime
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
        
    return series


def clean_price_anomalies(series: pd.Series, ticker: str) -> pd.Series:
    """
    Clean extreme price anomalies (bad ticks) by replacing them with NaN and forward filling.
    Uses a rolling median check where prices that deviate by a factor of 10x are flagged.
    """
    # Use rolling median of 5 days (centered)
    rolling_med = series.rolling(window=5, center=True, min_periods=1).median()
    ratio = series / rolling_med
    
    anomalies = (ratio < 0.1) | (ratio > 10.0)
    if anomalies.any():
        dates_anomalous = series.index[anomalies].strftime("%Y-%m-%d").tolist()
        print(f"[CLEANING] Ticker '{ticker}': Flagged and cleaned {len(dates_anomalous)} price anomalies on dates: {dates_anomalous}")
        cleaned_series = series.copy()
        cleaned_series[anomalies] = np.nan
        # Forward fill the NaN values
        cleaned_series = cleaned_series.ffill()
        return cleaned_series
    return series


def detect_and_flag_gaps(series: pd.Series, ticker: str) -> None:
    """
    Detect and log gaps of missing data (NaN) within the active trading range of the asset.
    Only checks for gaps between the first and last valid price index.
    """
    first_idx = series.first_valid_index()
    last_idx = series.last_valid_index()
    
    if first_idx is None or last_idx is None:
        print(f"Ticker '{ticker}' contains no valid data.")
        return
        
    # Slice to active range
    active_series = series.loc[first_idx:last_idx]
    is_nan = active_series.isna()
    
    if not is_nan.any():
        return
        
    # Group consecutive NaNs to find block lengths
    blocks = is_nan.ne(is_nan.shift()).cumsum()
    nan_blocks = is_nan.groupby(blocks)
    
    for block_id, block in nan_blocks:
        if block.iloc[0]:  # True indicates this block consists of NaNs
            gap_len = len(block)
            start_date = block.index[0].strftime("%Y-%m-%d")
            end_date = block.index[-1].strftime("%Y-%m-%d")
            if gap_len > 1:
                print(
                    f"[WARNING] Ticker '{ticker}': Gap of {gap_len} days detected "
                    f"from {start_date} to {end_date}. This exceeds the 1-day fill limit!"
                )
            else:
                print(f"[INFO] Ticker '{ticker}': Isolated 1-day gap detected on {start_date}.")


def load_and_align_data(
    tickers: List[str], start_date: str, end_date: str, cache_dir: str = "data/raw"
) -> pd.DataFrame:
    """
    Download/load tickers, clean anomalies, outer-join them, check and flag gaps, 
    forward-fill isolated 1-day gaps, and align all series.

    Parameters
    ----------
    tickers : List[str]
        List of tickers to process.
    start_date : str
        Start date.
    end_date : str
        End date.
    cache_dir : str
        Caching directory.

    Returns
    -------
    pd.DataFrame
        Aligned price DataFrame.
    """
    series_dict = {}
    for ticker in tickers:
        df = download_price_data(ticker, start_date, end_date, cache_dir=cache_dir)
        series = extract_price_series(df)
        series.name = ticker
        
        # Clean price anomalies (e.g., GOLDBEES bad tick in Dec 2019)
        series = clean_price_anomalies(series, ticker)
        
        series_dict[ticker] = series

    # Outer join to align all dates in a union index
    combined_df = pd.concat(series_dict.values(), axis=1, join="outer")
    combined_df.sort_index(inplace=True)

    # Detect gaps in each series prior to filling
    print("\n--- Gap Detection Analysis ---")
    for ticker in tickers:
        detect_and_flag_gaps(combined_df[ticker], ticker)

    # Forward fill only isolated single-day gaps (limit=1)
    filled_df = combined_df.ffill(limit=1)

    # Drop any remaining NaN rows
    aligned_df = filled_df.dropna()

    # Calculate row dropping statistics
    union_rows = len(combined_df)
    aligned_rows = len(aligned_df)
    
    # We can also compare against the primary equity benchmark if present
    equity_ticker = next((t for t in tickers if "NSEI" in t or "NIFTY" in t), tickers[0])
    equity_raw_len = combined_df[equity_ticker].dropna().shape[0]
    
    print("\n--- Alignment Statistics ---")
    print(f"Total dates in union of all series: {union_rows}")
    print(f"Total dates where ALL assets have active data: {aligned_rows}")
    print(f"Dropped rows from union of dates: {union_rows - aligned_rows} ({(union_rows - aligned_rows)/union_rows*100:.2f}%)")
    print(f"Equity benchmark ('{equity_ticker}') trading days: {equity_raw_len}")
    print(f"Dropped rows relative to equity benchmark calendar: {equity_raw_len - aligned_rows} ({(equity_raw_len - aligned_rows)/equity_raw_len*100:.2f}%)")
    print("-----------------------------\n")

    return aligned_df


def convert_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price DataFrame to daily log returns.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices.

    Returns
    -------
    pd.DataFrame
        DataFrame of daily log returns.
    """
    log_returns = np.log(prices).diff().dropna()
    return log_returns
