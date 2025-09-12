"""Data models for portfolio construction system.

This module defines the core data structures used throughout the portfolio
construction pipeline, including portfolios, signals, risk metrics, and
configuration objects.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class Signal:
    """Individual signal with confidence and metadata.

    Represents a single forecast μ for an asset with associated confidence,
    timing, and attribution metadata.

    Attributes:
        asset: Asset identifier (e.g., ETF ticker, sector name).
        mu: Expected return forecast (annualized).
        confidence: Signal confidence [0, 1] based on model agreement.
        half_life: Signal decay half-life in days.
        ic_estimate: Information coefficient estimate from backtesting.
        sleeve: Signal source identifier (e.g., "sector_committee", "momentum").
        timestamp: Signal generation timestamp.
        metadata: Additional signal-specific information.
    """

    asset: str
    mu: float
    confidence: float = 1.0
    half_life: int = 20
    ic_estimate: float = 0.05
    sleeve: str = "default"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate signal parameters."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.half_life <= 0:
            raise ValueError(f"Half-life must be positive, got {self.half_life}")


@dataclass
class RiskMetrics:
    """Portfolio risk and exposure metrics.

    Comprehensive risk statistics including concentration, correlation clustering,
    factor exposures, and operational metrics like turnover.

    Attributes:
        max_sector_concentration: Largest sector weight (absolute).
        max_asset_concentration: Largest individual asset weight (absolute).
        gross_exposure: Sum of absolute weights.
        net_exposure: Long weights minus short weights.
        estimated_beta: Portfolio beta vs. market benchmark.
        estimated_volatility: Annualized portfolio volatility estimate.
        correlation_cluster_exposure: Exposure to correlated asset clusters.
        factor_exposures: Exposures to style factors (value, momentum, etc.).
        turnover_estimate: Estimated portfolio turnover vs. previous period.
        leverage_adjusted_notional: Notional adjusted for leveraged instruments.
        concentration_violations: List of assets violating concentration limits.
        liquidity_risk: Liquidity-weighted risk estimate.
    """

    max_sector_concentration: float = 0.0
    max_asset_concentration: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    estimated_beta: float = 0.0
    estimated_volatility: float = 0.0
    correlation_cluster_exposure: float = 0.0
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    turnover_estimate: float = 0.0
    leverage_adjusted_notional: float = 0.0
    concentration_violations: List[str] = field(default_factory=list)
    liquidity_risk: float = 0.0


@dataclass
class Portfolio:
    """Complete portfolio specification with allocations and metadata.

    Contains all information needed to implement a portfolio including
    long/short positions, hedge positions, risk metrics, and audit trail.

    Attributes:
        long_positions: ETF ticker -> weight for long positions.
        short_positions: Inverse ETF ticker -> weight for short positions.
        hedge_positions: Market hedge (SPDN/SH) ticker -> weight.
        signals: List of input signals used in construction.
        risk_metrics: Computed portfolio risk statistics.
        optimization_method: Optimizer used ("mvo", "risk_parity", etc.).
        constraints_applied: List of constraint names that were active.
        construction_cost: Estimated cost of portfolio construction.
        timestamp: Portfolio construction timestamp.
        vintage: Rebalancing vintage identifier (for staggered ensemble).
        config: Configuration used for construction.
        attribution: Performance attribution by signal/sleeve.
        metadata: Additional portfolio-specific information.
    """

    long_positions: Dict[str, float] = field(default_factory=dict)
    short_positions: Dict[str, float] = field(default_factory=dict)
    hedge_positions: Dict[str, float] = field(default_factory=dict)
    signals: List[Signal] = field(default_factory=list)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    optimization_method: str = "mvo"
    constraints_applied: List[str] = field(default_factory=list)
    construction_cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    vintage: Optional[str] = None
    config: Optional["PortfolioConfig"] = None
    attribution: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_positions(self) -> Dict[str, float]:
        """Combined dictionary of all positions."""
        positions = {}
        positions.update(self.long_positions)
        positions.update(self.short_positions)
        positions.update(self.hedge_positions)
        return positions

    @property
    def total_weight(self) -> float:
        """Sum of all position weights (should be ~1.0 for fully invested)."""
        return sum(self.all_positions.values())

    def get_sector_exposure(self, sector_mappings: Dict[str, str]) -> Dict[str, float]:
        """Calculate net sector exposures.

        Args:
            sector_mappings: ETF ticker -> sector name mapping.

        Returns:
            Dictionary of sector name -> net exposure.
        """
        sector_exposures = {}
        for ticker, weight in self.all_positions.items():
            sector = sector_mappings.get(ticker, "Unknown")
            sector_exposures[sector] = sector_exposures.get(sector, 0.0) + weight
        return sector_exposures


@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction.

    Contains all parameters needed to construct a portfolio including
    risk limits, optimization settings, and operational constraints.

    Attributes:
        max_sector_weight: Maximum weight per sector.
        max_asset_weight: Maximum weight per individual asset.
        max_gross_exposure: Maximum sum of absolute weights.
        max_net_exposure: Maximum net long/short bias.
        target_beta: Target portfolio beta.
        beta_tolerance: Tolerance around target beta.
        rebalance_threshold: Minimum position change to trigger trade.
        max_monthly_turnover: Maximum monthly turnover budget.
        min_holding_period: Minimum holding period in days.
        max_cluster_weight: Maximum weight in correlated clusters.
        cluster_definitions: Asset -> cluster mappings.
        cost_model: Transaction cost model parameters.
        risk_model: Risk model parameters.
    """

    max_sector_weight: float = 0.30
    max_asset_weight: float = 0.20
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 0.20
    target_beta: float = 0.0
    beta_tolerance: float = 0.1
    rebalance_threshold: float = 0.002
    max_monthly_turnover: float = 0.20
    min_holding_period: int = 7
    max_cluster_weight: float = 0.50
    cluster_definitions: Dict[str, str] = field(default_factory=dict)
    cost_model: Dict[str, Any] = field(default_factory=dict)
    risk_model: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization.

    Parameters controlling the optimization process including
    optimizer selection, penalty terms, and convergence criteria.

    Attributes:
        method: Optimization method ("mvo", "penalized_mvo", "risk_parity", "black_litterman").
        risk_aversion: Risk aversion parameter for MVO.
        turnover_penalty: Penalty for portfolio turnover.
        concentration_penalty: Penalty for concentrated positions.
        leverage_penalty: Penalty for leverage usage.
        l1_penalty: L1 regularization strength.
        l2_penalty: L2 regularization strength.
        shrinkage_intensity: Covariance shrinkage intensity [0, 1].
        max_iterations: Maximum optimization iterations.
        convergence_tolerance: Convergence tolerance.
        solver: Optimization solver ("cvxpy", "scipy", "custom").
        warm_start: Whether to use previous solution as starting point.
    """

    method: str = "mvo"
    risk_aversion: float = 1.0
    turnover_penalty: float = 0.001
    concentration_penalty: float = 0.01
    leverage_penalty: float = 0.001
    l1_penalty: float = 0.0
    l2_penalty: float = 0.001
    shrinkage_intensity: float = 0.5
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    solver: str = "cvxpy"
    warm_start: bool = True


@dataclass
class RebalancingConfig:
    """Configuration for portfolio rebalancing.

    Parameters controlling the rebalancing process including
    scheduling, vintage management, and execution settings.

    Attributes:
        core_frequency: Core rebalancing frequency ("monthly", "weekly").
        ensemble_vintages: Number of overlapping vintage portfolios.
        vintage_weight: Weight per vintage (1/ensemble_vintages).
        patch_frequency: Fast signal patch frequency ("daily", "intraday").
        execution_window: Trading window constraints.
        child_order_size: Maximum child order size for large trades.
        venue_preferences: Trading venue preferences by asset class.
        latency_tolerance: Maximum execution latency tolerance.
    """

    core_frequency: str = "monthly"
    ensemble_vintages: int = 4
    vintage_weight: float = 0.25
    patch_frequency: str = "weekly"
    execution_window: Dict[str, Any] = field(default_factory=dict)
    child_order_size: float = 100000.0  # Maximum child order size
    venue_preferences: Dict[str, str] = field(default_factory=dict)
    latency_tolerance: float = 30.0  # Seconds
