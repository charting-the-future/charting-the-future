"""Test suite for sector committee analysis system.

This package contains comprehensive tests for Phase 1 (Deep Research Scoring System)
including unit tests, integration tests, performance validation, and compliance testing.

Test Structure:
- unit/: Unit tests for individual modules (90%+ coverage target)
- integration/: End-to-end workflow validation
- performance/: Latency, cost, and throughput testing
- fixtures/: Shared test data and mock objects
- conftest.py: Pytest configuration and fixtures

Test Categories:
- Functional: Core scoring logic and validation
- Performance: <5 min latency, <$5 cost targets
- Compliance: Schema validation, audit trails
- Error Handling: Timeout, API failures, validation errors
"""

from typing import Dict, Any

# Test configuration
TEST_CONFIG = {
    "timeout_seconds": 60,  # Shorter timeout for tests
    "test_sectors": ["Information Technology", "Financials", "Health Care"],
    "performance_targets": {
        "max_latency_ms": 300000,  # 5 minutes
        "max_cost_usd": 5.0,
        "min_confidence": 0.3,
        "min_consensus": 0.8,
    },
}


# Test utilities
def create_mock_sector_rating() -> Dict[str, Any]:
    """Create a mock sector rating for testing."""
    return {
        "rating": 4,
        "summary": "Bullish outlook based on strong fundamentals and positive sentiment trends.",
        "sub_scores": {"fundamentals": 4, "sentiment": 4, "technicals": 3},
        "weights": {"fundamentals": 0.5, "sentiment": 0.3, "technicals": 0.2},
        "weighted_score": 3.8,
        "rationale": [
            {
                "pillar": "fundamentals",
                "reason": "Strong earnings growth and reasonable valuations",
                "impact": "positive",
                "confidence": 0.8,
            }
        ],
        "references": [
            {
                "url": "https://example.com/analysis",
                "title": "Sector Analysis Report",
                "description": "Comprehensive sector analysis with key metrics",
                "accessed_at": "2025-09-08T18:00:00Z",
                "accessible": True,
            }
        ],
        "confidence": 0.75,
    }


def assert_valid_sector_rating(rating: Dict[str, Any]) -> None:
    """Assert that a sector rating is valid."""
    # Check required fields
    required_fields = [
        "rating",
        "summary",
        "sub_scores",
        "weights",
        "weighted_score",
        "rationale",
        "references",
        "confidence",
    ]
    for field in required_fields:
        assert field in rating, f"Missing required field: {field}"

    # Check value ranges
    assert 1 <= rating["rating"] <= 5, f"Invalid rating: {rating['rating']}"
    assert 0.0 <= rating["confidence"] <= 1.0, (
        f"Invalid confidence: {rating['confidence']}"
    )

    # Check sub-scores
    for pillar in ["fundamentals", "sentiment", "technicals"]:
        score = rating["sub_scores"][pillar]
        assert 1 <= score <= 5, f"Invalid {pillar} score: {score}"

    # Check weights sum to 1.0
    weight_sum = sum(rating["weights"].values())
    assert abs(weight_sum - 1.0) < 0.01, f"Weights don't sum to 1.0: {weight_sum}"


__all__ = ["TEST_CONFIG", "create_mock_sector_rating", "assert_valid_sector_rating"]
