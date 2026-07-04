"""
Optimizer Module
Implements portfolio optimization strategies (e.g., mean-variance, min-variance,
and regime-aware asset allocation) using cvxpy.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, Tuple, Optional, List, Union


def optimize_portfolio(
    expected_returns: Union[np.ndarray, pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    regime: str,
    asset_names: Optional[List[str]] = None,
    max_weight_crisis: float = 0.5,
    target_return: Optional[float] = None,
) -> Dict[str, float]:
    """
    Perform regime-specific portfolio optimization using cvxpy.

    Objectives per regime:
    - Bull: Minimize variance subject to a target return constraint (convex QP formulation).
      Justification: Maximizing the Sharpe ratio directly is a non-convex fractional program.
      While solvable via the Stoy-Dufour transformation, it is mathematically unstable if
      expected returns are negative. Minimizing variance subject to a target return (w^T * mu >= R)
      is a standard convex Quadratic Program (QP) that is highly stable, guarantees global convergence,
      and behaves cleanly under cvxpy.
    - Bear: Minimize portfolio volatility (minimum variance).
    - Crisis: Minimize portfolio volatility (minimum variance) AND cap single-asset weights
      at max_weight_crisis (e.g., 50%) to prevent degenerate corner allocations.

    Constraints for all regimes:
    - Long-only: w >= 0 (no short selling).
    - Fully invested: sum(w) = 1.
    - Custom upper bound per asset.

    Parameters
    ----------
    expected_returns : Union[np.ndarray, pd.Series]
        Expected daily (or annualized) returns for each asset.
    cov_matrix : Union[np.ndarray, pd.DataFrame]
        Covariance matrix of daily (or annualized) asset returns.
    regime : str
        The market regime ('Bull', 'Bear', 'Crisis').
    asset_names : Optional[List[str]], default None
        List of asset tickers/names. If expected_returns is a pd.Series, its index is used.
    max_weight_crisis : float, default 0.5
        Maximum weight allowed for any single asset during a Crisis regime.
    target_return : Optional[float], default None
        Target return for the Bull regime. If None, defaults to the average of expected returns.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping asset names to their optimal portfolio weights.
    """
    # 1. Align and convert inputs to numpy arrays
    if isinstance(expected_returns, pd.Series):
        if asset_names is None:
            asset_names = expected_returns.index.tolist()
        expected_returns = expected_returns.values
        
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
        
    n_assets = len(expected_returns)
    
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
    # Ensure inputs are floats
    expected_returns = np.array(expected_returns, dtype=float)
    cov_matrix = np.array(cov_matrix, dtype=float)
    
    # 2. Set up optimization variables
    w = cp.Variable(n_assets)
    
    # Common constraints
    constraints = [
        w >= 0,            # No shorting
        cp.sum(w) == 1     # Fully invested
    ]
    
    # 3. Setup regime-specific objectives and additional constraints
    if regime == "Bull":
        # Target Return constraint
        if target_return is None:
            # Feasible fallback: mean return of the assets
            target_return = np.mean(expected_returns)
            
        # Minimize portfolio variance: w^T * Sigma * w
        # subject to: expected return >= target_return
        # If the target return is mathematically infeasible (higher than the max expected return),
        # we relax it to the mean of expected returns
        max_possible_return = np.max(expected_returns)
        if target_return > max_possible_return:
            target_return = np.mean(expected_returns)
            
        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        constraints.append(w @ expected_returns >= target_return)
        
    elif regime == "Bear":
        # Minimize portfolio variance: w^T * Sigma * w
        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        
    elif regime == "Crisis":
        # Minimize portfolio variance: w^T * Sigma * w
        # Additional constraint: cap max single-asset weight
        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        constraints.append(w <= max_weight_crisis)
        
    else:
        # Default fallback: Global Minimum Variance
        print(f"[WARNING] Unknown regime '{regime}' specified. Falling back to Minimum Variance.")
        objective = cp.Minimize(cp.quad_form(w, cov_matrix))

    # 4. Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP)  # Use OSQP solver (highly stable for QP)
    except Exception as e:
        print(f"[ERROR] Solver failed: {e}. Attempting fallback solver...")
        problem.solve(solver=cp.CLARABEL)
        
    # 5. Check if solved successfully
    if problem.status not in ["optimal", "feasible"]:
        print(f"[WARNING] Optimization status is '{problem.status}'. Falling back to equal weights.")
        optimal_weights = np.ones(n_assets) / n_assets
    else:
        optimal_weights = w.value
        # Clean up numerical noise (ensure w >= 0 and sum(w) = 1)
        optimal_weights = np.clip(optimal_weights, 0, 1)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
    # 6. Map results to dictionary
    weight_dict = {asset_names[i]: float(optimal_weights[i]) for i in range(n_assets)}
    return weight_dict
