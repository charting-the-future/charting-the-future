"""Configuration for portfolio construction system.

This module defines ETF mappings, risk parameters, and default configurations
used throughout the portfolio construction pipeline.
"""

# Core ETF Mappings for 11 SPDR Sectors
SECTOR_LONG_ETF = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Information Technology": "XLK",
}

# Inverse ETF Mappings with leverage ratios
SECTOR_INVERSE_ETF = {
    "Communication Services": ("CCON", -2.0),  # ProShares UltraShort
    "Consumer Discretionary": ("SCC", -2.0),  # ProShares UltraShort
    "Consumer Staples": ("SZK", -2.0),  # ProShares UltraShort
    "Energy": ("DUG", -2.0),  # ProShares UltraShort Oil & Gas
    "Financials": ("SEF", -1.0),  # 1× inverse available
    "Health Care": ("RXD", -2.0),  # ProShares UltraShort Health Care
    "Industrials": ("SIJ", -2.0),  # ProShares UltraShort Industrials
    "Materials": ("SMN", -2.0),  # ProShares UltraShort Materials
    "Real Estate": ("REK", -1.0),  # 1× inverse available
    "Utilities": ("SDP", -2.0),  # ProShares UltraShort Utilities
    "Information Technology": ("REW", -2.0),  # ProShares UltraShort Technology
}

# Beta Hedge Options for Market Neutralization
BROAD_HEDGE_ETF = {
    "SPY": "SPDN",  # Direxion Daily S&P 500 Bear 1X Shares
    "SPX": "SH",  # ProShares Short S&P 500
    "QQQ": "PSQ",  # ProShares Short QQQ
}

# Default Risk Parameters
DEFAULT_RISK_PARAMS = {
    "max_sector_weight": 0.30,  # 30% maximum per sector
    "max_asset_weight": 0.20,  # 20% maximum per individual asset
    "max_gross_exposure": 1.0,  # 100% gross budget
    "max_net_exposure": 0.20,  # 20% maximum net long/short bias
    "target_beta": 0.0,  # Beta-neutral target
    "beta_tolerance": 0.1,  # ±10bps beta tolerance
    "rebalance_threshold": 0.002,  # 20bps minimum trade size
    "max_monthly_turnover": 0.20,  # 20% monthly turnover budget
    "min_holding_period": 7,  # 7-day minimum hold (avoid wash sales)
    "max_cluster_weight": 0.50,  # 50% maximum for correlated cluster
}

# Correlation Clustering Definitions
CORRELATION_CLUSTERS = {
    # Technology cluster (highly correlated during market stress)
    "tech_cluster": [
        "Information Technology",
        "Communication Services",
        "Consumer Discretionary",
    ],
    # Defensive cluster
    "defensive_cluster": ["Utilities", "Consumer Staples", "Health Care"],
    # Cyclical cluster
    "cyclical_cluster": ["Industrials", "Materials", "Energy"],
    # Financial cluster
    "financial_cluster": ["Financials", "Real Estate"],
}

# ETF Ticker to Sector Mapping (for reverse lookup)
TICKER_TO_SECTOR = {}
for sector, ticker in SECTOR_LONG_ETF.items():
    TICKER_TO_SECTOR[ticker] = sector
for sector, (ticker, _) in SECTOR_INVERSE_ETF.items():
    TICKER_TO_SECTOR[ticker] = sector

# Default Optimization Parameters
DEFAULT_OPTIMIZATION_PARAMS = {
    "method": "mvo",  # Mean-variance optimization
    "risk_aversion": 1.0,  # Risk aversion parameter
    "turnover_penalty": 0.001,  # Turnover penalty coefficient
    "concentration_penalty": 0.01,  # Concentration penalty coefficient
    "leverage_penalty": 0.001,  # Leverage penalty coefficient
    "l1_penalty": 0.0,  # L1 regularization (sparsity)
    "l2_penalty": 0.001,  # L2 regularization (smoothness)
    "shrinkage_intensity": 0.5,  # Covariance shrinkage [0, 1]
    "max_iterations": 1000,  # Maximum optimization iterations
    "convergence_tolerance": 1e-6,  # Convergence tolerance
    "solver": "cvxpy",  # Optimization solver
    "warm_start": True,  # Use previous solution as starting point
}

# Default Transaction Cost Model
DEFAULT_COST_MODEL = {
    "commission_per_trade": 0.0,  # Commission per trade (assume zero)
    "spread_bps": {  # Bid-ask spread by asset class
        "sector_etf": 2.0,  # 2bps for SPDR sector ETFs
        "inverse_etf": 5.0,  # 5bps for inverse ETFs
        "broad_market": 1.0,  # 1bps for broad market ETFs
    },
    "impact_model": "linear",  # Impact model type
    "impact_coefficient": 0.1,  # Impact coefficient (bps per $1M)
    "permanent_impact_ratio": 0.5,  # Permanent impact as % of total
    "minimum_trade_size": 1000.0,  # Minimum trade size ($)
}

# Default Rebalancing Configuration
DEFAULT_REBALANCING_CONFIG = {
    "core_frequency": "monthly",  # Core rebalancing frequency
    "ensemble_vintages": 4,  # Number of overlapping vintages
    "vintage_weight": 0.25,  # Weight per vintage (1/4)
    "patch_frequency": "weekly",  # Fast signal patch frequency
    "execution_window": {  # Trading window constraints
        "market_open_buffer": 30,  # Minutes after market open
        "market_close_buffer": 30,  # Minutes before market close
        "avoid_earnings": True,  # Avoid trading around earnings
        "avoid_opex": True,  # Avoid options expiration
    },
    "child_order_size": 100000.0,  # Maximum child order size ($)
    "latency_tolerance": 30.0,  # Maximum execution latency (seconds)
}

# Factor Model Definitions
FACTOR_DEFINITIONS = {
    "market": {
        "description": "Broad market exposure",
        "benchmark": "SPY",
        "loading_method": "regression",
    },
    "value": {
        "description": "Value factor exposure",
        "benchmark": "IWD",  # iShares Russell 1000 Value ETF
        "loading_method": "regression",
    },
    "momentum": {
        "description": "Momentum factor exposure",
        "benchmark": "MTUM",  # iShares MSCI USA Momentum Factor ETF
        "loading_method": "regression",
    },
    "quality": {
        "description": "Quality factor exposure",
        "benchmark": "QUAL",  # iShares MSCI USA Quality Factor ETF
        "loading_method": "regression",
    },
    "size": {
        "description": "Size factor exposure",
        "benchmark": "IWM",  # iShares Russell 2000 ETF
        "loading_method": "regression",
    },
    "low_volatility": {
        "description": "Low volatility factor exposure",
        "benchmark": "USMV",  # iShares MSCI USA Min Vol Factor ETF
        "loading_method": "regression",
    },
}

# Risk Model Parameters
DEFAULT_RISK_MODEL_PARAMS = {
    "covariance_method": "shrinkage",  # Covariance estimation method
    "shrinkage_target": "identity",  # Shrinkage target
    "shrinkage_intensity": 0.5,  # Shrinkage intensity
    "factor_model": True,  # Use factor model decomposition
    "factor_count": 6,  # Number of factors to use
    "residual_risk_floor": 0.01,  # Minimum residual risk
    "half_life_days": 60,  # Risk model half-life
    "min_observations": 250,  # Minimum observations for estimation
    "outlier_threshold": 3.0,  # Z-score threshold for outliers
    "volatility_regime_adjustment": True,  # Adjust for volatility regimes
}

# Signal Processing Parameters
DEFAULT_SIGNAL_PARAMS = {
    "calibration_method": "ic_scaling",  # Signal calibration method
    "blending_method": "ic_proportional",  # Multi-sleeve blending method
    "confidence_weighting": True,  # Use confidence weighting
    "orthogonalization": True,  # Orthogonalize to factors
    "decay_method": "exponential",  # Signal decay method
    "min_confidence": 0.1,  # Minimum signal confidence
    "max_leverage": 2.0,  # Maximum signal leverage
    "regime_awareness": True,  # Adjust for market regimes
}

# Monitoring and Attribution Parameters
DEFAULT_MONITORING_PARAMS = {
    "attribution_frequency": "daily",  # Attribution calculation frequency
    "exposure_monitoring": True,  # Monitor factor exposures
    "correlation_monitoring": True,  # Monitor correlation drift
    "capacity_monitoring": True,  # Monitor capacity constraints
    "performance_benchmark": "SPY",  # Performance benchmark
    "risk_benchmark": "SPY",  # Risk benchmark
    "attribution_window": 30,  # Attribution window (days)
    "alert_thresholds": {  # Alert thresholds
        "beta_deviation": 0.15,  # Beta deviation alert
        "concentration_violation": 0.35,  # Concentration violation alert
        "turnover_excess": 0.25,  # Excess turnover alert
        "tracking_error": 0.02,  # Tracking error alert
    },
}

# All default configurations combined
DEFAULT_CONFIGS = {
    "risk": DEFAULT_RISK_PARAMS,
    "optimization": DEFAULT_OPTIMIZATION_PARAMS,
    "cost_model": DEFAULT_COST_MODEL,
    "rebalancing": DEFAULT_REBALANCING_CONFIG,
    "risk_model": DEFAULT_RISK_MODEL_PARAMS,
    "signals": DEFAULT_SIGNAL_PARAMS,
    "monitoring": DEFAULT_MONITORING_PARAMS,
}
