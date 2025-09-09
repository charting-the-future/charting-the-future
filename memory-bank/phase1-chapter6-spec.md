# Phase 1 Project Specification: Chapter 6 Deep Research Scoring System

## Project Overview

**Phase:** 1 of 2  
**Chapter:** 6 - Multi-Agent Sector Analysis  
**Timeline:** 2 weeks  
**Dependencies:** None (uses existing deep_research_runner.py)

## Scope Definition

### In Scope
- Multi-agent sector analysis for 11 SPDR ETF sectors
- 1-5 numerical score generation with tri-pillar analysis
- JSON schema enforcement and validation
- Multi-model ensemble aggregation (OpenAI o4-mini + o3-deep-research)
- Complete audit trail with references and rationale
- Performance optimization for <5 minute analysis latency

### Out of Scope
- Portfolio construction or position sizing
- ETF allocation or beta neutralization
- Trading signals or execution logic
- Risk management beyond confidence scoring
- Multi-provider support beyond OpenAI (future phase)

### Boundaries
- **Input:** Sector name + analysis parameters
- **Output:** 1-5 score + structured audit trail
- **Hard Stop:** NO portfolio construction logic

## Technical Specifications

### Core Interfaces

```python
# Primary Interface
class SectorAgent:
    def analyze_sector(
        self, 
        sector_name: str, 
        horizon_weeks: int = 4,
        weights_hint: Optional[Dict[str, float]] = None
    ) -> SectorRating:
        """Analyze sector and return 1-5 rating with audit trail."""
        pass

# Data Models
class SectorRating(TypedDict):
    rating: int                               # 1-5 final score (REQUIRED)
    summary: str                             # Concise explanation
    sub_scores: Dict[str, int]               # fundamentals|sentiment|technicals
    weights: Dict[str, float]                # pillar weights used
    weighted_score: float                    # computed roll-up
    rationale: List[Dict[str, object]]       # supporting reasons
    references: List[Dict[str, str]]         # cited sources
    confidence: float                        # 0-1 reliability estimate

# Factory Pattern
class ModelFactory:
    @staticmethod
    def create(model: ModelName) -> ResearchClient:
        """Create appropriate research client for model type."""
        pass

# Ensemble Pattern
class EnsembleAggregator:
    def aggregate_results(
        self, 
        model_results: List[ModelResult]
    ) -> SectorRating:
        """Combine multiple model outputs into consensus rating."""
        pass
```

### JSON Schema Enforcement

```json
{
  "type": "object",
  "properties": {
    "rating": {"type": "integer", "minimum": 1, "maximum": 5},
    "summary": {"type": "string", "minLength": 10, "maxLength": 500},
    "sub_scores": {
      "type": "object",
      "properties": {
        "fundamentals": {"type": "integer", "minimum": 1, "maximum": 5},
        "sentiment": {"type": "integer", "minimum": 1, "maximum": 5},
        "technicals": {"type": "integer", "minimum": 1, "maximum": 5}
      },
      "required": ["fundamentals", "sentiment", "technicals"],
      "additionalProperties": false
    },
    "weights": {
      "type": "object",
      "properties": {
        "fundamentals": {"type": "number", "minimum": 0, "maximum": 1},
        "sentiment": {"type": "number", "minimum": 0, "maximum": 1},
        "technicals": {"type": "number", "minimum": 0, "maximum": 1}
      },
      "required": ["fundamentals", "sentiment", "technicals"]
    },
    "weighted_score": {"type": "number", "minimum": 1, "maximum": 5},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "required": ["rating", "summary", "sub_scores", "weights", "weighted_score", "rationale", "references", "confidence"],
  "additionalProperties": false
}
```

### Package Structure

```
sector_committee/scoring/
├── __init__.py              # Exports: SectorAgent, SectorRating
├── agents.py                # SectorAgent implementation
├── models.py                # Data models and TypedDict definitions
├── factory.py               # ModelFactory and ResearchClient interfaces
├── ensemble.py              # EnsembleAggregator implementation
├── schema.py                # JSON schema validation
└── audit.py                 # Audit trail logging utilities
```

## Deliverables

### D1: Core Sector Agent Implementation
- **File:** `sector_committee/scoring/agents.py`
- **Description:** Main SectorAgent class with analyze_sector method
- **Integration:** Uses existing deep_research_runner.py patterns
- **Testing:** Unit tests with 90%+ coverage

### D2: Data Models and Schema Validation
- **File:** `sector_committee/scoring/models.py`
- **Description:** SectorRating TypedDict and validation logic
- **Schema:** JSON schema enforcement with 100% compliance
- **Testing:** Schema validation tests for all edge cases

### D3: Model Factory and Ensemble System
- **Files:** `sector_committee/scoring/factory.py`, `sector_committee/scoring/ensemble.py`
- **Description:** Factory pattern for model creation, ensemble aggregation
- **Models:** OpenAI o4-mini-deep-research + o3-deep-research support
- **Testing:** Multi-model consensus validation

### D4: Audit Trail System
- **File:** `sector_committee/scoring/audit.py`
- **Description:** Complete audit logging with timestamps and compliance metadata
- **Format:** JSON files with full request/response tracking
- **Testing:** Audit completeness validation

### D5: Integration Tests
- **File:** `tests/test_phase1_integration.py`
- **Description:** End-to-end scoring workflow validation
- **Coverage:** All 11 SPDR sectors analyzed successfully
- **Performance:** <5 minute latency verification

## Acceptance Criteria

### AC1: Functional Requirements
- ✅ **Sector Coverage:** All 11 SPDR sectors (XLB, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY, XLC) analyzed successfully
- ✅ **Score Generation:** Valid 1-5 ratings produced for 100% of analysis attempts
- ✅ **Schema Compliance:** 100% of outputs pass JSON schema validation
- ✅ **Tri-Pillar Analysis:** Fundamentals, sentiment, and technicals scores generated for each sector

### AC2: Performance Requirements
- ✅ **Latency:** <5 minutes average response time per sector analysis
- ✅ **Reliability:** 95%+ successful score generation rate
- ✅ **Consistency:** Ensemble models agree within 1 point 80%+ of the time
- ✅ **Cost Efficiency:** <$5 inference cost per sector analysis

### AC3: Quality Requirements
- ✅ **Unit Test Coverage:** 90%+ code coverage for all scoring modules
- ✅ **Integration Testing:** End-to-end workflow passes for all sectors
- ✅ **Audit Compliance:** 100% of analyses have complete reference trails
- ✅ **Error Handling:** Graceful degradation when individual models fail

### AC4: Technical Requirements
- ✅ **API Integration:** OpenAI o4-mini and o3-deep-research models working
- ✅ **JSON Validation:** Schema enforcement prevents malformed outputs
- ✅ **Ensemble Logic:** Multi-model aggregation mathematically correct
- ✅ **Logging System:** Complete audit trail for compliance review

## Success Metrics

### Primary KPIs
- **Information Quality:** 60%+ sector calls correct over 4-week horizon
- **Model Agreement:** <1 point average deviation between ensemble models
- **Operational Uptime:** 99%+ availability during market hours
- **Cost Control:** Total inference cost <$55 for full 11-sector analysis

### Secondary KPIs
- **Reference Quality:** 80%+ of URLs accessible and relevant
- **Confidence Calibration:** ±10% accuracy between stated and realized confidence
- **Processing Speed:** Full 11-sector sweep completed in <30 minutes
- **API Reliability:** <5% timeout/error rate for model requests

## Risk Mitigation

### Technical Risks
- **API Rate Limits:** Implement exponential backoff and queue management
- **Model Timeouts:** Set 300-second timeout with graceful fallback
- **Schema Failures:** Comprehensive validation with error reporting
- **Cost Overruns:** Budget alerts at $3 per analysis threshold

### Operational Risks
- **Data Quality:** URL accessibility validation and error handling
- **Model Drift:** Regular ensemble agreement monitoring
- **Performance Degradation:** Latency alerts and performance tracking
- **Compliance Gaps:** Audit trail completeness verification

## Testing Strategy

### Unit Tests (90% Coverage Target)
```python
# tests/scoring/
├── test_agents.py           # SectorAgent functionality
├── test_models.py           # Data model validation  
├── test_factory.py          # ModelFactory patterns
├── test_ensemble.py         # Aggregation logic
├── test_schema.py           # JSON validation
└── test_audit.py            # Logging functionality
```

### Integration Tests
```python
# tests/integration/
├── test_end_to_end.py       # Full scoring workflow
├── test_multi_model.py      # Ensemble execution
├── test_performance.py      # Latency and cost validation
└── test_compliance.py       # Audit trail verification
```

### Performance Tests
- **Load Testing:** 11 sectors analyzed concurrently
- **Stress Testing:** API timeout and retry scenarios
- **Cost Testing:** Inference expense tracking and alerts
- **Latency Testing:** Response time distribution analysis

## Definition of Done

Phase 1 is complete when:

1. ✅ All 11 SPDR sectors can be analyzed with valid 1-5 scores
2. ✅ 95%+ reliability rate achieved across 100 test analyses
3. ✅ Ensemble models agree within 1 point 80%+ of the time
4. ✅ Complete audit trail generated for compliance review
5. ✅ Unit test coverage exceeds 90% with all tests passing
6. ✅ Integration tests validate end-to-end scoring workflow
7. ✅ Cost efficiency under $5 per sector analysis verified
8. ✅ Performance target of <5 minutes per analysis met
9. ✅ JSON schema validation achieves 100% compliance
10. ✅ Documentation complete and memory bank updated

**Gate Criteria:** Phase 2 cannot begin until all acceptance criteria are met and validated through automated testing.
