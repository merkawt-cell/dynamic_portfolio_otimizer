"""
Portfolio Optimization Module

Implements various portfolio optimization strategies:
- Maximum Sharpe Ratio
- Minimum Volatility
- Risk Parity
- Target Return
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional, Literal


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """Calculate expected portfolio return."""
    return float(weights @ mu)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """Calculate portfolio volatility (standard deviation)."""
    return float(np.sqrt(weights @ cov @ weights))


def portfolio_sharpe(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, risk_free: float) -> float:
    """Calculate portfolio Sharpe ratio."""
    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov)
    return (ret - risk_free) / vol if vol > 0 else 0.0


def negative_sharpe(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, risk_free: float) -> float:
    """Negative Sharpe ratio for minimization."""
    return -portfolio_sharpe(weights, mu, cov, risk_free)


def optimize_max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free: float = 0.0,
    allow_short: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Optimize portfolio for maximum Sharpe ratio.

    Args:
        mu: Expected returns vector
        cov: Covariance matrix
        risk_free: Risk-free rate
        allow_short: Allow short selling (negative weights)

    Returns:
        Tuple of (weights, metrics)
    """
    n = len(mu)

    # Initial guess: equal weights
    w0 = np.ones(n) / n

    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Bounds: 0 to 1 for each weight (no short selling by default)
    if allow_short:
        bounds = tuple((-1, 1) for _ in range(n))
    else:
        bounds = tuple((0, 1) for _ in range(n))

    # Optimize
    result = minimize(
        negative_sharpe,
        w0,
        args=(mu, cov, risk_free),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    weights = result.x

    # Clean up very small weights
    weights[np.abs(weights) < 1e-6] = 0
    weights = weights / np.sum(weights)  # Renormalize

    metrics = {
        "expected_return": portfolio_return(weights, mu),
        "volatility": portfolio_volatility(weights, cov),
        "sharpe": portfolio_sharpe(weights, mu, cov, risk_free)
    }

    return weights, metrics


def optimize_min_volatility(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free: float = 0.0,
    allow_short: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Optimize portfolio for minimum volatility.

    Args:
        mu: Expected returns vector
        cov: Covariance matrix
        risk_free: Risk-free rate
        allow_short: Allow short selling

    Returns:
        Tuple of (weights, metrics)
    """
    n = len(mu)

    w0 = np.ones(n) / n

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    if allow_short:
        bounds = tuple((-1, 1) for _ in range(n))
    else:
        bounds = tuple((0, 1) for _ in range(n))

    result = minimize(
        lambda w: portfolio_volatility(w, cov),
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    weights = result.x
    weights[np.abs(weights) < 1e-6] = 0
    weights = weights / np.sum(weights)

    metrics = {
        "expected_return": portfolio_return(weights, mu),
        "volatility": portfolio_volatility(weights, cov),
        "sharpe": portfolio_sharpe(weights, mu, cov, risk_free)
    }

    return weights, metrics


def optimize_target_return(
    mu: np.ndarray,
    cov: np.ndarray,
    target_return: float,
    risk_free: float = 0.0,
    allow_short: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Optimize portfolio for minimum volatility at a target return.

    Args:
        mu: Expected returns vector
        cov: Covariance matrix
        target_return: Target portfolio return
        risk_free: Risk-free rate
        allow_short: Allow short selling

    Returns:
        Tuple of (weights, metrics)
    """
    n = len(mu)

    w0 = np.ones(n) / n

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: portfolio_return(w, mu) - target_return}
    ]

    if allow_short:
        bounds = tuple((-1, 1) for _ in range(n))
    else:
        bounds = tuple((0, 1) for _ in range(n))

    result = minimize(
        lambda w: portfolio_volatility(w, cov),
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    weights = result.x
    weights[np.abs(weights) < 1e-6] = 0
    weights = weights / np.sum(weights)

    metrics = {
        "expected_return": portfolio_return(weights, mu),
        "volatility": portfolio_volatility(weights, cov),
        "sharpe": portfolio_sharpe(weights, mu, cov, risk_free)
    }

    return weights, metrics


def optimize_risk_parity(
    cov: np.ndarray,
    mu: Optional[np.ndarray] = None,
    risk_free: float = 0.0
) -> Tuple[np.ndarray, Dict]:
    """
    Risk Parity optimization - equal risk contribution from each asset.

    Args:
        cov: Covariance matrix
        mu: Expected returns vector (optional, for metrics only)
        risk_free: Risk-free rate

    Returns:
        Tuple of (weights, metrics)
    """
    n = cov.shape[0]

    def risk_contribution(weights):
        """Calculate risk contribution of each asset."""
        port_vol = portfolio_volatility(weights, cov)
        marginal_contrib = cov @ weights
        risk_contrib = weights * marginal_contrib / port_vol
        return risk_contrib

    def risk_parity_objective(weights):
        """Objective: minimize squared differences in risk contributions."""
        rc = risk_contribution(weights)
        target_rc = 1.0 / n  # Equal risk contribution
        return np.sum((rc - target_rc) ** 2)

    w0 = np.ones(n) / n

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0.01, 1) for _ in range(n))  # Min 1% in each asset

    result = minimize(
        risk_parity_objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    weights = result.x
    weights = weights / np.sum(weights)

    if mu is None:
        mu = np.zeros(n)

    metrics = {
        "expected_return": portfolio_return(weights, mu),
        "volatility": portfolio_volatility(weights, cov),
        "sharpe": portfolio_sharpe(weights, mu, cov, risk_free)
    }

    return weights, metrics


def optimize_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free: float = 0.0,
    strategy: Literal["max_sharpe", "min_volatility", "risk_parity", "equal_weight"] = "max_sharpe",
    target_return: Optional[float] = None,
    allow_short: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Main portfolio optimization function.

    Args:
        mu: Expected returns vector
        cov: Covariance matrix
        risk_free: Risk-free rate (annualized)
        strategy: Optimization strategy
        target_return: Target return (for target_return strategy)
        allow_short: Allow short selling

    Returns:
        Tuple of (weights array, metrics dict)
    """
    if strategy == "max_sharpe":
        return optimize_max_sharpe(mu, cov, risk_free, allow_short)

    elif strategy == "min_volatility":
        return optimize_min_volatility(mu, cov, risk_free, allow_short)

    elif strategy == "risk_parity":
        return optimize_risk_parity(cov, mu, risk_free)

    elif strategy == "target_return" and target_return is not None:
        return optimize_target_return(mu, cov, target_return, risk_free, allow_short)

    else:  # equal_weight fallback
        n = len(mu)
        weights = np.ones(n) / n
        metrics = {
            "expected_return": portfolio_return(weights, mu),
            "volatility": portfolio_volatility(weights, cov),
            "sharpe": portfolio_sharpe(weights, mu, cov, risk_free)
        }
        return weights, metrics


def compute_efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free: float = 0.0,
    n_points: int = 50
) -> Dict:
    """
    Compute the efficient frontier.

    Args:
        mu: Expected returns vector
        cov: Covariance matrix
        risk_free: Risk-free rate
        n_points: Number of points on the frontier

    Returns:
        Dictionary with frontier data
    """
    # Get min and max return portfolios
    _, min_vol_metrics = optimize_min_volatility(mu, cov, risk_free)
    _, max_sharpe_metrics = optimize_max_sharpe(mu, cov, risk_free)

    min_ret = min_vol_metrics["expected_return"]
    max_ret = max(mu)  # Maximum single asset return

    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_returns = []
    frontier_volatilities = []
    frontier_sharpes = []

    for target in target_returns:
        try:
            _, metrics = optimize_target_return(mu, cov, target, risk_free)
            frontier_returns.append(metrics["expected_return"])
            frontier_volatilities.append(metrics["volatility"])
            frontier_sharpes.append(metrics["sharpe"])
        except:
            continue

    return {
        "returns": frontier_returns,
        "volatilities": frontier_volatilities,
        "sharpes": frontier_sharpes,
        "max_sharpe_portfolio": max_sharpe_metrics,
        "min_volatility_portfolio": min_vol_metrics
    }
