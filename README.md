# Regime-Switching Portfolio Optimization

A quantitative research project to model market regimes (using Hidden Markov Models or similar approaches) and optimize asset allocation dynamically based on the current regime.

## Project Structure

```text
regime-portfolio/
  data/               # Cached price data (gitignored)
  src/
    data_loader.py    # Download and cache price data
    features.py       # Feature engineering for regime detection
    regime_model.py   # Regime-switching models (e.g., HMM)
    optimizer.py      # Portfolio optimization under different regimes
    backtest.py       # Strategy backtesting engine
    validation.py     # Validation and performance evaluation metrics
  notebooks/
    01_data_exploration.ipynb
    02_full_pipeline.ipynb
  tests/              # Unit tests
  README.md
  requirements.txt
  .gitignore
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
