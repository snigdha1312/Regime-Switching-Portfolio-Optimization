"""
Unit tests for the optimizer module.
Verifies that the regime-specific portfolio optimization solves correctly.
"""

import numpy as np
import pytest
from src.optimizer import optimize_portfolio


def test_crisis_regime_optimization():
    """
    Test that the Crisis regime optimizer:
    1. Caps individual asset weights at the specified threshold (e.g. 50%).
    2. Tilts weights toward the lowest-volatility asset.
    3. Respects budget and no-shorting constraints.
    """
    # 3 assets: A (low vol), B (medium vol), C (high vol)
    asset_names = ["Asset_A", "Asset_B", "Asset_C"]
    
    # Volatilities: A = 10%, B = 20%, C = 30%
    # Covariance: diagonal (no correlation)
    cov = np.diag([0.10**2, 0.20**2, 0.30**2])
    
    # Expected returns (not directly used for Bear/Crisis objectives, but required input)
    expected_returns = np.array([0.05, 0.10, 0.15])
    
    # 1. Run Crisis optimization with a 50% cap
    max_cap = 0.50
    weights = optimize_portfolio(
        expected_returns=expected_returns,
        cov_matrix=cov,
        regime="Crisis",
        asset_names=asset_names,
        max_weight_crisis=max_cap
    )
    
    # Verify outputs
    assert len(weights) == 3
    assert set(weights.keys()) == set(asset_names)
    
    # Check constraints
    sum_w = sum(weights.values())
    assert sum_w == pytest.approx(1.0, abs=1e-6)
    
    for asset, weight in weights.items():
        assert weight >= 0.0, f"Negative weight found for {asset}: {weight}"
        assert weight <= max_cap + 1e-6, f"Weight of {asset} ({weight}) exceeded cap of {max_cap}"
        
    # Check logic: Asset_A is the lowest-vol asset, so it should be capped at 50%
    assert weights["Asset_A"] == pytest.approx(0.50, abs=1e-4)
    
    # Asset_B should have higher weight than Asset_C since it has lower vol
    assert weights["Asset_B"] > weights["Asset_C"]
    
    # Verify the exact min-variance solution under cap constraint:
    # Since Asset_A is capped at 0.5, the remaining 0.5 weight must be split between B and C
    # to minimize variance:
    # Min w_B^2 * 0.2^2 + w_C^2 * 0.3^2  subject to w_B + w_C = 0.5
    # w_B = 0.5 * (0.3^2) / (0.2^2 + 0.3^2) = 0.5 * 0.09 / 0.13 = 0.34615
    # w_C = 0.5 * (0.2^2) / (0.2^2 + 0.3^2) = 0.5 * 0.04 / 0.13 = 0.15384
    assert weights["Asset_B"] == pytest.approx(0.34615, abs=1e-3)
    assert weights["Asset_C"] == pytest.approx(0.15384, abs=1e-3)
    
    print("Crisis regime optimization unit test passed successfully.")
