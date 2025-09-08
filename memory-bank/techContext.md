# Technical Context: Implementation Stack and Integration

## Development Environment

### Python Environment
- **Python Version**: 3.13 (target version for modern type hints and performance)
- **Package Manager**: `uv` for fast dependency management and environment isolation
- **Project Structure**: `pyproject.toml` based configuration with MIT licensing

### Key Dependencies
```toml
[project]
dependencies = [
    "openai>=1.50.0",           # OpenAI API with structured outputs support
    "python-dotenv>=1.0.0",     # Environment variable management
    "pytest>=8.4.2",            # Testing framework
    "ruff>=0.12.12",            # Linting and formatting
    "pydantic>=2.0.0",          # Data validation and serialization
    "numpy>=1.24.0",            # Numerical computations
    "pandas>=2.0.0",            # Data manipulation for portfolio construction
]
```

### Development Commands
```bash
# Environment setup
uv sync                          # Install dependencies
uv run python -m sector_committee # Run package

# Code quality
uv run ruff check .             # Linting
uv run ruff format .            # Code formatting
uv run pytest tests/           # Run test suite
```

## OpenAI Integration Architecture

### API Configuration
```python
# Environment variables required
OPENAI_API_KEY=sk-...                    # OpenAI API key
DEEP_RESEARCH_LOG_DIR=./logs/deep_research # Audit log directory
```

### Model Specifications
**o4-mini-deep-research-2025-06-26:**
- **Use Case**: Fast, cost-effective sector analysis
- **Features**: Web search integration, structured outputs
- **Latency**: ~10-30 seconds per sector
- **Cost**: ~$0.10-0.50 per sector analysis

**o3-deep-research-2025-06-26:**
- **Use Case**: Deep reasoning with enhanced research capabilities
- **Features**: Advanced web search, complex reasoning chains
- **Latency**: ~60-180 seconds per sector  
- **Cost**: ~$1.00-3.00 per sector analysis

### Structured Output Implementation
```python
# JSON Schema Enforcement via OpenAI Structured Outputs
response = client.responses.create(
    model=self.model_id,
    input=[{"role": "developer", "content": content_blocks}],
    tools=[{"type": "web_search_preview"}],  # Enable web browsing
    response_format={
        "type": "json_schema", 
        "json_schema": {
            "name": "sector_rating", 
            "schema": self._schema()
        }
    },
)
```

### Web Search Integration
```python
# Automatic reference downloading and citation
tools=[{"type": "web_search_preview"}]
```
- **Capability**: Automatic URL access and content extraction
- **Sources**: Public financial data providers, SEC filings, news outlets
- **Validation**: URL accessibility and content relevance checking
- **Storage**: Full reference text downloaded for audit compliance

## Data Processing Pipeline

### Request/Response Flow
```python
# Input standardization
@dataclass(frozen=True)
class SectorRequest:
    sector: str                              # One of 11 SPDR sectors
    horizon_weeks: int = 4                   # Investment timeframe
    weights_hint: Optional[Dict[str, float]] # Pillar weight overrides

# Output validation
class SectorRating(TypedDict):
    rating: int                              # 1-5 validated range
    sub_scores: Dict[str, int]               # Pillar scores (1-5)
    weights: Dict[str, float]                # Applied weights (sum=1.0)
    weighted_score: float                    # Computed aggregate
    confidence: float                        # 0-1 reliability
    rationale: List[Dict[str, object]]       # Evidence chain
    references: List[Dict[str, str]]         # Downloadable sources
```

### ETF Universe Management
```python
# Sector mappings with operational metadata
SECTOR_LONG_ETF = {
    "Information Technology": "XLK",    # Primary long position
    "Financials": "XLF",               # Direct sector exposure
    # ... 9 additional SPDR sectors
}

SECTOR_INVERSE = {
    "Information Technology": ("REW", -2.0),  # 2x inverse ETF
    "Financials": ("SEF", -1.0),             # 1x inverse available
    # ... leverage-adjusted short positions
}
```

## Audit and Logging Infrastructure

### Comprehensive Audit Trail
```python
def write_audit_log(model: ModelName, req: SectorRequest, result: SectorRating) -> Path:
    ts = _now_utc_str()  # UTC timestamp for global consistency
    payload = {
        "timestamp_utc": ts,
        "model": model.name,
        "request": {
            "sector": req.sector,
            "horizon_weeks": req.horizon_weeks,
            "weights_hint": req.weights_hint
        },
        "result": result,  # Complete JSON response
        "compliance": {
            "prompt_hash": hash_prompt(req),      # Prompt versioning
            "schema_version": "v1.0",             # Schema evolution tracking
            "api_latency_ms": response_time,      # Performance monitoring
            "reference_count": len(result["references"])
        }
    }
```

### Log File Structure
```
logs/deep_research/
├── 20250908T151030Z_OPENAI_O4_MINI_DEEP_RESEARCH_Financials.json
├── 20250908T151045Z_OPENAI_O3_DEEP_RESEARCH_Financials.json
└── ensemble_20250908T151100Z_Financials.json
```

### Compliance Requirements
- **SR 11-7 Model Governance**: Full model decision reconstruction
- **MiFID II**: Trade rationale and evidence documentation  
- **Audit Readiness**: JSON logs with schema validation
- **Performance Attribution**: Model-level P&L contribution tracking

## Error Handling and Resilience

### API Failure Management
```python
# Exponential backoff for rate limits
@retry(exponential_backoff(base=2, max_retries=3))
def _call_openai_api(self, request):
    try:
        return self.client.responses.create(...)
    except RateLimitError:
        # Log and retry with backoff
    except TimeoutError:
        # Fallback to cached scores if available
    except ValidationError:
        # Schema mismatch handling
```

### Quality Gates
- **Schema Validation**: Strict JSON structure enforcement
- **Confidence Thresholds**: Minimum 0.3 confidence for order generation  
- **Cross-Model Consistency**: Alert if models disagree by >2 rating points
- **Reference Quality**: Minimum 3 accessible URLs per analysis

## Performance and Cost Monitoring

### Key Metrics Tracking
```python
# Real-time monitoring
class PerformanceMetrics:
    inference_latency_ms: float      # Response time by model
    api_cost_usd: float             # Cost per sector analysis
    reference_download_success: float # URL accessibility rate
    ensemble_agreement: float        # Model consensus level
    confidence_distribution: Dict    # Confidence histogram
```

### Cost Management
- **Daily Budgets**: $100/day inference limit with alerts
- **Model Selection**: Auto-fallback to cheaper models during high usage
- **Batch Processing**: Off-peak analysis for non-urgent updates
- **Cache Strategy**: 4-hour TTL for repeated sector requests

## Integration Points

### Portfolio Construction Interface
```python
# Score aggregation for portfolio manager
def aggregate_sector_scores(sector_ratings: List[SectorRating]) -> PortfolioTilts:
    tilts = {}
    for rating in sector_ratings:
        # Map 1-5 scores to -2 to +2 tilts
        tilt = (rating["rating"] - 3) * rating["confidence"]
        tilts[rating["sector"]] = tilt
    return normalize_to_gross_budget(tilts, target_gross=1.0)
```

### Risk Management Hooks
```python
# Real-time position monitoring
class RiskMonitor:
    def validate_portfolio(self, tilts: PortfolioTilts) -> RiskReport:
        # Sector concentration limits
        # Beta neutrality verification  
        # Correlation clustering checks
        # Turnover budget compliance
```

### Order Management System (OMS) Integration
- **Order Translation**: Tilts → sized positions → executable orders
- **Venue Routing**: Direct ETF trades vs. inverse ETF alternatives
- **Execution Timing**: Batch orders during liquid market hours
- **Settlement Tracking**: T+2 settlement for ETF trades

This technical foundation supports systematic, scalable, and auditable multi-agent investment decision-making with institutional-grade operational controls and regulatory compliance.
