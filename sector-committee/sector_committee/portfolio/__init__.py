"""Portfolio Construction Pipeline for Chapter 7.

This module provides a comprehensive portfolio construction framework that converts
sector scores into beta-neutral ETF allocations with sophisticated risk management.

Main Components:
- PortfolioConstructor: Main interface for building portfolios
- Portfolio, Signal, RiskMetrics: Core data models
- Signal processing: Calibration, blending, confidence weighting
- Risk management: Shrinkage, factor models, concentration limits
- Optimization: MVO, penalized MVO, Risk Parity, Black-Litterman
- Execution: Rebalancing, cost modeling, turnover control
- Monitoring: Attribution, exposure tracking, model hygiene

Usage:
    from sector_committee.portfolio import PortfolioConstructor, Portfolio

    constructor = PortfolioConstructor()
    sector_scores = {"Information Technology": 5, "Energy": 2, ...}
    portfolio = constructor.build_portfolio(sector_scores)
"""

from .models import (
    Portfolio,
    Signal,
    RiskMetrics,
    PortfolioConfig,
    OptimizationConfig,
    RebalancingConfig,
)
from .constructor import PortfolioConstructor
from .config import (
    SECTOR_LONG_ETF,
    SECTOR_INVERSE_ETF,
    BROAD_HEDGE_ETF,
    DEFAULT_RISK_PARAMS,
    DEFAULT_OPTIMIZATION_PARAMS,
)

__all__ = [
    # Main classes
    "PortfolioConstructor",
    # Data models
    "Portfolio",
    "Signal",
    "RiskMetrics",
    "PortfolioConfig",
    "OptimizationConfig",
    "RebalancingConfig",
    # Configuration
    "SECTOR_LONG_ETF",
    "SECTOR_INVERSE_ETF",
    "BROAD_HEDGE_ETF",
    "DEFAULT_RISK_PARAMS",
    "DEFAULT_OPTIMIZATION_PARAMS",
]
