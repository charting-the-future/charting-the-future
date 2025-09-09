# Phase 2 Project Specification: Chapter 7 Portfolio Construction Pipeline

## Project Overview

**Phase:** 2 of 2  
**Chapter:** 7 - Portfolio Construction and Risk Management  
**Timeline:** 2 weeks  
**Dependencies:** Phase 1 completion (requires working sector scores)

## Scope Definition

### In Scope
- Portfolio construction from 11 sector scores (1-5 ratings)
- Score-to-tilt mapping: {1,2,3,4,5} → {-2,-1,0,+1,+2}
- Beta neutralization using inverse ETFs and broad market hedges
- Position sizing with concentration limits and turnover controls
- ETF allocation system for 11 SPDR sectors + inverse mappings
- Risk management framework with real-time monitoring

### Out of Scope
- Sector score generation (handled in Phase 1)
- Multi-agent analysis or LLM inference
- Trade execution or order management systems
- Real-time market data feeds
- Performance attribution beyond basic P&L tracking

### Boundaries
- **Input:** Dictionary of 11 sector scores (1-5) from Phase 1
- **Output:** ETF allocations with beta-neutral positioning
- **Hard Stop:** NO score generation - pure portfolio construction

## Technical Specifications

### Core Interfaces

```python
# Primary Interface
class PortfolioConstructor:
    def build_portfolio(
        self,
        sector_scores: Dict[str, int],
        gross_target: float = 1.0,
        max_sector_weight: float = 0.30
    ) -> Portfolio:
        """Convert sector scores to beta-neutral ETF allocations."""
        pass

# Data Models
@dataclass
class Portfolio:
    long_positions: Dict[str, float]          # ETF ticker -> weight
    short_positions: Dict[str, float]         # Inverse ETF ticker -> weight
    hedge_positions: Dict[str, float]         # Market hedge (SPDN/SH) -> weight
    gross_exposure: float                     # Total absolute exposure
    net_exposure: float                       # Long - short exposure
    estimated_beta: float                     # Portfolio beta vs SPY
    sector_tilts: Dict[str, float]           # Sector -> tilt (-2 to +2)
    risk_metrics: RiskMetrics                 # Concentration, turnover stats
    timestamp: datetime                       # Construction timestamp

@dataclass
class RiskMetrics:
    max_sector_concentration: float           # Largest sector weight
    turnover_estimate: float                  # Estimated turnover vs previous
    correlation_cluster_exposure: float       # Tech cluster concentration
    leverage_adjusted_notional: float         # Accounting for 2x inverse ETFs

# Score Mapping
class ScoreMapper:
    @staticmethod
    def scores_to_tilts(sector_scores: Dict[str, int]) -> Dict[str, float]:
        """Map 1-5 scores to -2 to +2 portfolio tilts."""
        pass

# Beta Hedging
class BetaHedger:
    def calculate_hedge_ratio(
        self,
        long_positions: Dict[str, float],
        short_positions: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate SPDN/SH hedge to neutralize beta."""
        pass
```

### ETF Configuration

```python
# Core ETF Mappings
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

# Inverse ETF Mappings (with leverage ratios)
SECTOR_INVERSE = {
    "Communication Services": ("CCON", -2.0),  # ProShares UltraShort
    "Consumer Discretionary": ("SCC", -2.0),
    "Consumer Staples": ("SZK", -2.0),
    "Energy": ("DUG", -2.0),
    "Financials": ("SEF", -1.0),              # 1× inverse available
    "Health Care": ("RXD", -2.0),
    "Industrials": ("SIJ", -2.0),
    "Materials": ("SMN", -2.0),
    "Real Estate": ("REK", -1.0),             # 1× inverse available
    "Utilities": ("SDP", -2.0),
    "Information Technology": ("REW", -2.0),
}

# Beta Hedge Options
BROAD_HEDGE = {
    "SPY": "SPDN",  # -1x S&P 500
    "SPX": "SH",    # -1x S&P 500 alternative
}
```

### Risk Parameters

```python
# Portfolio Constraints
MAX_SECTOR_WEIGHT = 0.30           # 30% maximum per sector
MAX_GROSS_EXPOSURE = 1.0           # 100% gross budget
MAX_NET_EXPOSURE = 0.20            # 20% maximum net long/short bias
TARGET_BETA = 0.0                  # Beta-neutral target
BETA_TOLERANCE = 0.1               # ±10bps beta tolerance

# Turnover Controls
REBALANCE_THRESHOLD = 0.002        # 20bps minimum trade size
MAX_MONTHLY_TURNOVER = 0.20        # 20% monthly turnover budget
MIN_HOLDING_PERIOD = 7             # 7-day minimum hold (avoid wash sales)

# Correlation Clustering
TECH_CLUSTER = ["Information Technology", "Communication Services", "Consumer Discretionary"]
MAX_CLUSTER_WEIGHT = 0.50          # 50% maximum for correlated cluster
```

### Package Structure

```
sector_committee/portfolio/
├── __init__.py              # Exports: PortfolioConstructor, Portfolio
├── constructor.py           # PortfolioConstructor implementation
├── models.py                # Portfolio and RiskMetrics dataclasses
├── mapping.py               # ScoreMapper score-to-tilt conversion
├── hedging.py               # BetaHedger neutralization logic
├── risk.py                  # Risk management and constraint enforcement
├── config.py                # ETF mappings and risk parameters
└── utils.py                 # Portfolio utilities and calculations
```

## Deliverables

### D1: Portfolio Construction Engine
- **File:** `sector_committee/portfolio/constructor.py`
- **Description:** Main PortfolioConstructor class with build_portfolio method
- **Features:** Score-to-tilt mapping, position sizing, constraint enforcement
- **Testing:** Unit tests with mathematical validation

### D2: Score Mapping System
- **File:** `sector_committee/portfolio/mapping.py`
- **Description:** ScoreMapper converting 1-5 scores to -2/+2 tilts
- **Algorithm:** Linear mapping with confidence weighting
- **Testing:** Edge case validation for all score combinations

### D3: Beta Neutralization System
- **File:** `sector_committee/portfolio/hedging.py`
- **Description:** BetaHedger calculating SPDN/SH hedge ratios
- **Logic:** Residual beta calculation and neutralization
- **Testing:** Beta precision validation within ±0.1 tolerance

### D4: Risk Management Framework
- **File:** `sector_committee/portfolio/risk.py`
- **Description:** Concentration limits, turnover controls, correlation clustering
- **Constraints:** 30% sector caps, 50% cluster limits, 20% turnover budget
- **Testing:** Constraint violation detection and handling

### D5: ETF Configuration System
- **File:** `sector_committee/portfolio/config.py`
- **Description:** Complete ETF mappings with leverage adjustments
- **Coverage:** 11 SPDR sectors + inverse funds + broad market hedges
- **Testing:** Ticker validation and leverage ratio verification

## Acceptance Criteria

### AC1: Portfolio Construction Requirements
- ✅ **Score Processing:** Accept all 11 sector scores (1-5) and convert to portfolio
- ✅ **Tilt Mapping:** Accurate conversion {1,2,3,4,5} → {-2,-1,0,+1,+2}
- ✅ **Position Sizing:** Gross exposure hitting 100% ±2% target
- ✅ **ETF Allocation:** Valid allocations for all 11 SPDR sectors + inverses

### AC2: Risk Management Requirements
- ✅ **Beta Control:** |portfolio beta| < 0.1 for 100 random score combinations
- ✅ **Concentration:** No sector exceeding 30% allocation
- ✅ **Cluster Control:** Tech cluster (XLK+XLC+XLY) max 50% combined
- ✅ **Turnover Management:** Rebalance threshold 20bps minimum trade size

### AC3: Performance Requirements
- ✅ **Latency:** Full portfolio construction completing within 30 seconds
- ✅ **Accuracy:** Beta calculations within ±0.05 of target
- ✅ **Reliability:** 100% successful portfolio generation for valid inputs
- ✅ **Efficiency:** Optimal position sizing minimizing tracking error

### AC4: Technical Requirements
- ✅ **ETF Validation:** All tickers verified as tradeable instruments
- ✅ **Leverage Handling:** -2x inverse ETFs properly notional-adjusted
- ✅ **Data Models:** Portfolio and RiskMetrics dataclasses complete
- ✅ **Error Handling:** Graceful handling of constraint violations

## Success Metrics

### Primary KPIs
- **Risk Precision:** Beta maintained within ±0.1 target 95%+ of time
- **Capital Efficiency:** Gross exposure = 100% ±2% for all constructions
- **Constraint Compliance:** 0% concentration limit violations
- **Performance Speed:** <30 seconds for full portfolio construction

### Secondary KPIs
- **Turnover Efficiency:** Trades triggered only when >20bps threshold
- **Hedge Effectiveness:** Residual beta after hedging <0.05 absolute
- **Correlation Management:** Tech cluster exposure <50% at all times
- **Operational Reliability:** 100% successful constructions for valid inputs

## Risk Mitigation

### Technical Risks
- **Inverse ETF Tracking:** Daily reset monitoring and path dependency alerts
- **Liquidity Constraints:** Position size limits based on average daily volume
- **Leverage Miscalculation:** Comprehensive testing of -2x notional adjustments
- **Beta Drift:** Real-time beta monitoring with rehedging triggers

### Operational Risks
- **Constraint Violations:** Automated checking and violation prevention
- **Market Microstructure:** Bid-ask spread impact on small positions
- **Correlation Breakdown:** Dynamic cluster monitoring and adjustment
- **Turnover Explosion:** Monthly turnover budget tracking and alerts

## Testing Strategy

### Unit Tests (95% Coverage Target)
```python
# tests/portfolio/
├── test_constructor.py      # Portfolio construction logic
├── test_mapping.py          # Score-to-tilt conversion
├── test_hedging.py          # Beta neutralization
├── test_risk.py             # Risk management constraints
├── test_config.py           # ETF mappings validation
└── test_utils.py            # Portfolio utilities
```

### Integration Tests
```python
# tests/integration/
├── test_end_to_end_portfolio.py   # Full construction workflow
├── test_beta_precision.py         # Beta neutralization accuracy
├── test_constraint_enforcement.py # Risk limit validation
└── test_performance.py            # Latency and efficiency
```

### Mathematical Validation Tests
- **Beta Calculations:** Monte Carlo testing with 1000 random portfolios
- **Tilt Mapping:** Linear relationship verification across all score ranges
- **Leverage Adjustments:** Inverse ETF notional calculations
- **Constraint Testing:** Boundary condition validation

### Scenario Tests
```python
# Extreme Score Scenarios
test_all_ones_scores()      # All sectors rated 1 (max short)
test_all_fives_scores()     # All sectors rated 5 (max long)
test_mixed_scores()         # Random combinations
test_tech_heavy_scores()    # High tech cluster scores

# Market Condition Tests
test_high_correlation()     # During market stress
test_low_volatility()       # During calm markets
test_sector_rotation()      # During style transitions
```

## Definition of Done

Phase 2 is complete when:

1. ✅ Portfolio construction handles all possible 11-sector score combinations
2. ✅ Beta control maintains |beta| < 0.1 for 95%+ of constructions
3. ✅ No concentration limits violated across 1000 test portfolios
4. ✅ All ETF mappings verified with valid, tradeable tickers
5. ✅ Performance target of <30 seconds per construction met
6. ✅ Unit test coverage exceeds 95% with all tests passing
7. ✅ Integration tests validate end-to-end portfolio workflow
8. ✅ Risk management constraints properly enforced
9. ✅ Leverage adjustments accurate for all inverse ETFs
10. ✅ Documentation complete and memory bank updated

**Completion Criteria:** Both Phase 1 and Phase 2 specifications fully implemented with all acceptance criteria met and validated through comprehensive testing.

## Integration with Phase 1

### Interface Contract
```python
# Phase 1 Output → Phase 2 Input
sector_scores: Dict[str, int] = {
    "Communication Services": 3,
    "Consumer Discretionary": 4,
    "Consumer Staples": 2,
    "Energy": 5,
    "Financials": 3,
    "Health Care": 4,
    "Industrials": 3,
    "Materials": 2,
    "Real Estate": 1,
    "Utilities": 2,
    "Information Technology": 5,
}

# Phase 2 Processing
constructor = PortfolioConstructor()
portfolio = constructor.build_portfolio(sector_scores)

# Final Output: Beta-neutral ETF allocations ready for execution
```

### End-to-End Validation
- Complete pipeline from sector analysis to tradeable positions
- Full audit trail from LLM reasoning to portfolio allocation
- Risk management integrated throughout the workflow
- Compliance documentation for regulatory review
