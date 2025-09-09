# Active Context: Spec-Driven Two-Phase Development

## Current Work: Implementing Spec-Driven Development Plan

We are implementing a **two-phase, spec-driven development approach** that clearly separates the book chapters and ensures measurable deliverables with defined acceptance criteria.

### Phase Structure Overview

**Phase 1 (Chapter 6): Deep Research Scoring System**
- **Scope:** Multi-agent sector analysis producing 1-5 scores only
- **Input:** Sector name + analysis parameters  
- **Output:** 1-5 numerical score + audit trail
- **Boundary:** NO portfolio construction - scores only

**Phase 2 (Chapter 7): Portfolio Construction Pipeline**
- **Scope:** Convert sector scores into beta-neutral portfolio positions
- **Input:** Dictionary of 11 sector scores (1-5)
- **Output:** ETF allocations with beta neutralization
- **Boundary:** NO score generation - portfolio construction only

### Current Implementation Foundation

**Existing Assets:**
- **Complete Algorithm**: `deep_research_runner.py` provides proven multi-agent scoring
- **Factory Pattern**: `ModelFactory` with OpenAI o4-mini + o3-deep-research support
- **JSON Schema**: Strict validation preventing parsing errors
- **Ensemble Logic**: Multi-model consensus with confidence weighting
- **Audit Trail**: Timestamped logs with complete request/response tracking

### Spec-Driven Development Structure

**Package Architecture:**
```
sector-committee/
├── sector_committee/
│   ├── scoring/          # Phase 1: Chapter 6 components
│   │   ├── agents.py     # SectorAgent implementation
│   │   ├── models.py     # Data models and schemas  
│   │   ├── factory.py    # ModelFactory patterns
│   │   └── ensemble.py   # Multi-model aggregation
│   ├── portfolio/        # Phase 2: Chapter 7 components
│   │   ├── constructor.py # PortfolioConstructor
│   │   ├── hedging.py    # Beta neutralization
│   │   └── risk.py       # Concentration & turnover controls
│   ├── common/           # Shared utilities
│   │   ├── config.py     # ETF mappings & parameters
│   │   └── utils.py      # Logging & audit utilities
│   └── __init__.py       # Package exports
├── tests/                # Spec-driven test suites
├── examples/             # Phase-specific demonstrations
└── configs/              # Configuration files
```

### Current Phase 1 Specifications

**Technical Interface:**
- `SectorAgent.analyze_sector(sector_name: str, horizon_weeks: int = 4) -> SectorRating`
- `SectorRating` TypedDict with exact schema compliance
- `ModelFactory` supporting current and future LLM providers
- `EnsembleAggregator` for multi-model consensus scoring

**Acceptance Criteria (Measurable):**
- ✅ Latency: <5 minutes per sector analysis
- ✅ Reliability: 95%+ successful score generation
- ✅ Consistency: Ensemble models agree within 1 point 80%+ of time  
- ✅ Unit test coverage: 90%+
- ✅ Schema validation: 100% pass rate for all 11 sectors

**Success Metrics:**
- Cost efficiency: <$5 inference cost per analysis
- Audit compliance: 100% complete reference trails
- Performance: Full 11-sector sweep in <30 minutes

### Current Phase 2 Specifications

**Technical Interface:**
- `PortfolioConstructor.build_portfolio(scores: Dict[str, int]) -> Portfolio`
- Score-to-tilt mapping: {1,2,3,4,5} → {-2,-1,0,+1,+2}
- `BetaHedger` for market neutralization using inverse ETFs + SPDN/SH
- Risk management with concentration limits and turnover controls

**Acceptance Criteria (Measurable):**
- ✅ Beta control: |portfolio beta| < 0.1 for 100 random score combinations
- ✅ Concentration: No sector >30% allocation  
- ✅ Capital efficiency: Gross exposure = 100% ±2%
- ✅ Performance: Full pipeline completion <30 seconds
- ✅ ETF mapping: Valid tickers for all 11 sectors + inverses

**Success Metrics:**
- Risk precision: Beta maintained within ±0.1 target 95%+ of time
- Turnover efficiency: Trades only when >20bps threshold exceeded
- Operational speed: Real-time portfolio updates during market hours

### Implementation Dependencies

**Phase 1 Dependencies:**
- ✅ Existing `deep_research_runner.py` implementation
- ✅ OpenAI API access (o4-mini + o3-deep-research)
- ✅ JSON schema definitions
- ✅ Audit logging requirements

**Phase 2 Dependencies:**
- ⏳ Phase 1 completion (sector scores as input)
- ✅ ETF mapping definitions (11 SPDR sectors + inverses)
- ✅ Risk parameter specifications
- ✅ Beta neutralization algorithms

**Critical Success Factors:**
- **Sequential Execution**: Phase 2 cannot begin until Phase 1 delivers working scores
- **Measurable Outcomes**: Every acceptance criterion must be 100% testable
- **Scope Discipline**: Strict boundaries prevent feature creep between phases
- **Compliance Focus**: Audit trail and risk controls built-in from start

### Next Implementation Steps

**Immediate Priority (Phase 1):**
1. **Package Structure**: Create `sector_committee/scoring/` modules
2. **Core Integration**: Adapt `deep_research_runner.py` into `SectorAgent` class
3. **Schema Enforcement**: Implement strict JSON validation with 100% compliance
4. **Testing Framework**: Build unit tests achieving 90%+ coverage
5. **Performance Validation**: Ensure <5 minute latency per sector analysis

**Phase 1 Completion Criteria:**
- All 11 SPDR sectors can be analyzed with valid 1-5 scores
- Ensemble models agree within 1 point 80%+ of the time
- Complete audit trail generated for compliance review
- Cost efficiency under $5 per sector analysis achieved
- Integration tests pass for end-to-end scoring workflow

**Phase 2 Preparation:**
- ETF mapping verification for all 11 sectors + inverse funds
- Risk parameter configuration (30% concentration limits, 20bps turnover threshold)
- Beta neutralization algorithm implementation and testing
- Portfolio construction mathematical validation

The spec-driven approach ensures every deliverable is measurable and testable before implementation begins, preventing scope creep and ensuring institutional-grade quality throughout the development process.
