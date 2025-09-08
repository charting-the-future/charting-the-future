# Active Context: Current Development Focus

## Current Work: Deep Research Runner Implementation

We are currently implementing the core sector analysis engine based on the `deep_research_runner.py` provided by the user. This represents the heart of the multi-agent system that converts LLM reasoning into structured sector scores.

### What We Have: Complete Implementation Framework

**Provided Code Structure:**
- **Factory Pattern**: `ModelFactory` creates appropriate `ResearchClient` instances
- **OpenAI Integration**: Support for o4-mini-deep-research and o3-deep-research models
- **Strict JSON Schema**: Enforced structured outputs preventing parsing errors
- **Ensemble Aggregation**: Multi-model scoring with confidence-weighted averaging
- **Comprehensive Audit Trail**: Timestamped logs with full request/response tracking

**Key Implementation Features:**
- **Web Search Integration**: Automatic reference downloading and citation
- **Tri-Pillar Scoring**: Structured analysis of fundamentals, sentiment, technicals
- **Configurable Weights**: Default 50/25/25 weighting with override capability
- **SOLID Architecture**: Clean separation of concerns, easily extensible

### Current Integration Challenge

The `deep_research_runner.py` exists as a standalone implementation, but we need to integrate it into the `charting-the-future/sector-committee/` package structure. The sector-committee directory currently contains empty subdirectories:

```
sector-committee/
├── sector_committee/     # Empty - needs core modules
├── notebooks/           # Empty - needs demonstration notebooks  
├── configs/             # Empty - needs ETF mappings and risk parameters
├── tests/               # Empty - needs unit and integration tests
└── examples/            # Empty - needs minimal usage scripts
```

### Immediate Next Steps

**1. Package Structure Setup**
- Create `sector_committee/__init__.py` with main exports
- Implement `sector_committee/agents.py` based on deep_research_runner patterns
- Build `sector_committee/portfolio.py` for score-to-position translation
- Add `sector_committee/hedging.py` for beta neutralization logic

**2. Configuration Management**
- Implement ETF mapping dictionaries (SECTOR_LONG_ETF, SECTOR_INVERSE)
- Create risk parameter configs (turnover bands, concentration limits)
- Build model configuration templates for different LLM providers

**3. Portfolio Construction Pipeline**
- Score aggregation from multiple sector agents
- Tilt mapping: {1,2,3,4,5} → {-2,-1,0,+1,+2}
- Position sizing with gross budget constraints
- Beta neutralization using broad market hedges (SPDN/SH)

### Current Design Decisions

**Model Selection:**
- Primary: OpenAI o4-mini-deep-research (speed/cost optimization)
- Secondary: OpenAI o3-deep-research (depth/accuracy validation)
- Future: Claude, Gemini, Grok for ensemble diversification

**ETF Implementation Strategy:**
- **Long Positions**: Direct SPDR sector ETFs (XLB, XLE, XLF, etc.)
- **Short Positions**: Mix of -1x and -2x inverse ETFs with notional adjustments
- **Beta Hedge**: SPDN or SH overlay sized to offset residual market exposure

**Operational Guardrails:**
- Weekly rebalance cadence aligned with signal half-life
- 20bps minimum trade threshold to prevent excessive turnover
- 30% maximum single-sector concentration
- Correlation-aware clustering constraints (Tech + Comm Services + Consumer Discretionary)

### Technical Integration Points

**Data Flow:**
1. `SectorRequest` → Multiple `ResearchClient` instances
2. Parallel model execution with web search and reference downloading
3. `ModelResult` aggregation into ensemble `SectorRating`
4. Score transformation to portfolio tilts with risk overlays
5. Order generation with venue routing and execution logic

**Error Handling:**
- Graceful degradation when individual models fail
- Fallback to cached scores when API limits are hit
- Confidence scaling when reference quality is poor
- Alert generation when model consensus breaks down

### Documentation and Testing Strategy

**Unit Tests Needed:**
- Schema validation for all JSON outputs
- Ensemble aggregation mathematical correctness
- ETF mapping and leverage calculations
- Portfolio construction edge cases

**Integration Tests Required:**
- End-to-end pipeline from sector request to orders
- Multi-model execution with timeout handling
- Reference downloading and validation
- Audit log completeness and format compliance

### Performance and Cost Monitoring

**Key Metrics to Track:**
- **Inference Latency**: Model response times by provider
- **API Costs**: Cost per sector analysis by model type
- **Reference Quality**: URL accessibility and content relevance
- **Ensemble Coherence**: Agreement levels between models

The immediate priority is translating the proven `deep_research_runner.py` patterns into a production-ready package that integrates seamlessly with the broader quantitative finance workflow described in the book chapters.
