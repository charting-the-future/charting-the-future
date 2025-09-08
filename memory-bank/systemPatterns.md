# System Patterns: Multi-Agent Architecture and Implementation

## Core Architectural Patterns

### 1. Factory Pattern for Model Clients

**Interface Definition:**
```python
class ResearchClient(abc.ABC):
    @abc.abstractmethod
    def run(self, req: SectorRequest) -> SectorRating:
        raise NotImplementedError
```

**Factory Implementation:**
```python
class ModelFactory:
    @staticmethod
    def create(model: ModelName) -> ResearchClient:
        if model in (ModelName.OPENAI_O4_MINI_DEEP_RESEARCH, ModelName.OPENAI_O3_DEEP_RESEARCH):
            return OpenAIDeepResearchClient(model)
        # Future: ClaudeResearchClient, GeminiResearchClient, GrokResearchClient
        raise ValueError(f"No client available for {model}")
```

**Benefits:**
- Open/Closed Principle: Add new models without modifying existing code
- Dependency Inversion: High-level orchestration depends on abstractions
- Testability: Mock clients for unit testing

### 2. Structured Data Models

**Request Structure:**
```python
@dataclass(frozen=True)
class SectorRequest:
    sector: str
    horizon_weeks: int = 4
    weights_hint: Optional[Dict[str, float]] = None
```

**Response Structure:**
```python
class SectorRating(TypedDict):
    rating: int                               # 1-5 final score
    summary: str                             # Concise explanation
    sub_scores: Dict[str, int]               # fundamentals|sentiment|technicals
    weights: Dict[str, float]                # pillar weights used
    weighted_score: float                    # computed roll-up
    rationale: List[Dict[str, object]]       # supporting reasons
    references: List[Dict[str, str]]         # cited sources
    confidence: float                        # 0-1 reliability estimate
```

### 3. ETF Universe and Mapping

**Long Position ETFs:**
```python
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
```

**Short Position ETFs (with leverage adjustments):**
```python
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
```

**Beta Hedge Options:**
```python
BROAD_HEDGE = {"SPY": "SPDN", "IVV": "SPDN", "VOO": "SPDN", "SPX": "SPDN"}
```

### 4. JSON Schema Enforcement

**Strict Output Validation:**
```python
{
    "type": "object",
    "properties": {
        "rating": {"type": "integer", "minimum": 1, "maximum": 5},
        "sub_scores": {
            "type": "object",
            "properties": {
                "fundamentals": {"type": "integer", "minimum": 1, "maximum": 5},
                "sentiment": {"type": "integer", "minimum": 1, "maximum": 5},
                "technicals": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["fundamentals", "sentiment", "technicals"],
            "additionalProperties": False,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["rating", "summary", "sub_scores", "weights", "weighted_score", "rationale", "references", "confidence"],
    "additionalProperties": False,
}
```

### 5. Ensemble Aggregation Pattern

**Multi-Model Scoring:**
```python
def aggregate_results(model_results: List[ModelResult]) -> SectorRating:
    # Average fundamental pillars
    fundamentals = _avg([mr.data["sub_scores"]["fundamentals"] for mr in model_results])
    sentiment = _avg([mr.data["sub_scores"]["sentiment"] for mr in model_results])
    technicals = _avg([mr.data["sub_scores"]["technicals"] for mr in model_results])
    
    # Reconcile rating vs weighted score
    weighted_avg = _avg([float(mr.data["weighted_score"]) for mr in model_results])
    rating_from_weighted = _map_weighted_to_rating(weighted_avg)
    
    return SectorRating(...)
```

**Rating Mapping Logic:**
```python
def _map_weighted_to_rating(weighted_score: float) -> int:
    if weighted_score < 1.5: return 1
    if weighted_score < 2.5: return 2
    if weighted_score < 3.5: return 3
    if weighted_score < 4.5: return 4
    return 5
```

### 6. Portfolio Construction Workflow

**Score to Tilt Mapping:**
```
Agent Scores: {1, 2, 3, 4, 5} → Portfolio Tilts: {-2, -1, 0, +1, +2}
```

**Position Sizing Algorithm:**
1. Aggregate committee scores per sector
2. Map to signed tilts with confidence weighting
3. Scale to gross budget constraint (∑|tilt| = 1.0)
4. Apply sector concentration caps (max 30% per sector)
5. Handle inverse ETF leverage adjustments
6. Compute residual beta and hedge with SPDN/SH

**Risk Control Framework:**
```python
# Rebalance Controls
REBALANCE_THRESHOLD = 0.002  # 20bps minimum trade size
TURNOVER_BUDGET = 0.20       # 20% monthly maximum
GROSS_EXPOSURE_LIMIT = 1.0   # 100% gross budget

# Concentration Limits
MAX_SECTOR_WEIGHT = 0.30     # 30% maximum per sector
MAX_SIDE_GROSS = 0.60        # 60% maximum long or short

# Correlation Clustering
CLUSTER_TECH = ["Information Technology", "Communication Services", "Consumer Discretionary"]
MAX_CLUSTER_WEIGHT = 0.50    # 50% maximum per correlated cluster
```

### 7. Audit Trail Pattern

**Request Logging:**
```python
def write_audit_log(model: ModelName, req: SectorRequest, result: SectorRating) -> Path:
    ts = _now_utc_str()
    payload = {
        "timestamp_utc": ts,
        "model": model.name,
        "request": {"sector": req.sector, "horizon_weeks": req.horizon_weeks},
        "result": result,
    }
    # Write to timestamped JSON file for compliance review
```

**Compliance Requirements:**
- Full reconstruction capability from logs to orders
- Prompt versioning and context hashing
- Reference URL download and validation
- Model attribution for P&L explanation

### 8. Error Handling and Resilience

**Graceful Degradation:**
- Individual model failures don't break ensemble
- API timeout handling with exponential backoff
- Cached score fallbacks during provider outages
- Confidence scaling based on data quality issues

**Quality Gates:**
- Schema validation before aggregation
- Cross-model consistency checks
- Reference URL accessibility validation
- Confidence threshold enforcement for order generation

### 9. Integration Patterns

**Tiered Pipeline Architecture:**
```
Fast Filter → Heavy Analyzer → Deterministic Trigger
```

**Data Flow Coordination:**
1. Market data ingestion and preprocessing
2. Parallel model execution with timeout management
3. Ensemble aggregation and quality validation
4. Portfolio construction with risk overlays
5. Order generation and execution routing

This architecture ensures systematic, auditable, and scalable multi-agent investment decision-making while maintaining the operational discipline required for production quantitative finance systems.
