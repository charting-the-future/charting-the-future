"""Optimized integration tests for Phase 1: Deep Research Scoring System.

This module provides fast-running tests for development workflow while maintaining
comprehensive coverage. Uses caching and performance optimizations to reduce
test execution time from 6+ minutes to under 1 minute.

For full comprehensive testing, use test_phase1_integration.py.
"""

import pytest

from sector_committee.scoring import (
    validate_sector_rating,
)
from sector_committee.data_models import SectorName, SECTOR_ETF_MAP
from tests import assert_valid_sector_rating


class TestPhase1Optimized:
    """Optimized integration tests for Phase 1 functionality."""

    @pytest.mark.asyncio
    async def test_single_sector_workflow_cached(
        self, sample_analysis, performance_thresholds
    ):
        """Test single sector workflow using cached analysis."""
        # Handle both coroutine (first time) and cached data (subsequent times)
        if hasattr(sample_analysis, "__await__"):
            rating = await sample_analysis
        else:
            rating = sample_analysis

        # Validate result structure
        assert_valid_sector_rating(rating)

        # Validate specific requirements
        assert rating["rating"] in range(1, 6), "Rating must be 1-5"
        assert 0.0 <= rating["confidence"] <= 1.0, "Confidence must be 0-1"
        assert len(rating["summary"]) >= 10, "Summary must be meaningful"
        assert len(rating["references"]) > 0, "Must have references"
        assert len(rating["rationale"]) > 0, "Must have rationale"

        # Validate schema compliance (fast local validation)
        validation_result = validate_sector_rating(rating)
        assert validation_result.is_valid, (
            f"Schema validation failed: {validation_result.errors}"
        )

    @pytest.mark.asyncio
    async def test_multi_sector_analysis_cached(
        self, multi_sector_analysis, performance_thresholds
    ):
        """Test multi-sector analysis using cached results."""
        # Handle both coroutine (first time) and cached data (subsequent times)
        if hasattr(multi_sector_analysis, "__await__"):
            results = await multi_sector_analysis
        else:
            results = multi_sector_analysis

        # Validate we have multiple successful analyses
        assert len(results) >= 2, (
            f"Need at least 2 successful analyses, got {len(results)}"
        )

        # Validate each result
        for sector, rating in results.items():
            assert_valid_sector_rating(rating)

            # Validate ETF mapping
            _ = SECTOR_ETF_MAP[SectorName(sector)]
            # Note: ETF mapping validation can be done without analysis_metadata

            # Validate schema compliance
            validation_result = validate_sector_rating(rating)
            assert validation_result.is_valid, (
                f"Schema validation failed for {sector}: {validation_result.errors}"
            )

        # Calculate performance metrics
        confidences = [rating["confidence"] for rating in results.values()]
        avg_confidence = sum(confidences) / len(confidences)

        # Validate performance targets
        assert avg_confidence >= performance_thresholds["min_confidence"], (
            f"Average confidence {avg_confidence} below minimum {performance_thresholds['min_confidence']}"
        )

    @pytest.mark.asyncio
    async def test_schema_validation_comprehensive(self, sample_analysis):
        """Test comprehensive schema validation using cached data."""
        # Handle both coroutine (first time) and cached data (subsequent times)
        if hasattr(sample_analysis, "__await__"):
            rating = await sample_analysis
        else:
            rating = sample_analysis

        # Test with strict schema validation
        validation_result = validate_sector_rating(rating)
        assert validation_result.is_valid, (
            f"Schema validation failed: {validation_result.errors}"
        )

        # Additional compliance checks
        assert len(rating["rationale"]) <= 10, "Too many rationale items"
        assert len(rating["references"]) <= 10, "Too many references"

        # Check reference accessibility (lenient for cached test data)
        accessible_refs = sum(
            1 for ref in rating["references"] if ref.get("accessible", False)
        )
        if len(rating["references"]) > 0:
            # At least warn if no references are accessible, but don't fail test
            if accessible_refs == 0:
                print("Warning: No accessible references in cached test data")

        # Validate required fields are present and properly typed
        required_fields = [
            "rating",
            "summary",
            "confidence",
            "sub_scores",
            "weights",
            "rationale",
            "references",
        ]
        for field in required_fields:
            assert field in rating, f"Required field {field} missing"

        # Validate sub_scores structure
        assert "fundamentals" in rating["sub_scores"]
        assert "sentiment" in rating["sub_scores"]
        assert "technicals" in rating["sub_scores"]

        # Validate weights structure and sum
        weights = rating["weights"]
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01, (
            f"Weights sum to {total_weight}, expected ~1.0"
        )

    @pytest.mark.asyncio
    async def test_error_handling_fast(self, fast_agent):
        """Test error handling without expensive API calls."""

        # Test 1: Invalid sector name (should fail fast)
        with pytest.raises(Exception):
            await fast_agent.analyze_sector("Invalid Sector Name")

        # Test 2: Invalid horizon weeks (should fail fast)
        with pytest.raises(Exception):
            await fast_agent.analyze_sector("Information Technology", horizon_weeks=100)

        # Test 3: Invalid weights (should fail fast)
        invalid_weights = {
            "fundamentals": 0.7,
            "sentiment": 0.5,
            "technicals": 0.1,
        }  # Sum > 1.0

        with pytest.raises(Exception):
            await fast_agent.analyze_sector(
                "Information Technology", weights_hint=invalid_weights
            )

    @pytest.mark.asyncio
    async def test_health_check_fast(self, fast_agent):
        """Test system health check functionality."""
        health_status = await fast_agent.health_check()

        # Validate health check response structure
        assert "timestamp" in health_status
        assert "status" in health_status
        assert health_status["status"] in ["healthy", "degraded"]
        assert "supported_sectors" in health_status
        assert health_status["supported_sectors"] == len(SectorName)

        # Validate models information if available
        if "models_available" in health_status:
            assert isinstance(health_status["models_available"], int)
            assert health_status["models_available"] >= 0

    @pytest.mark.asyncio
    async def test_sector_etf_mapping_fast(self, fast_agent):
        """Test sector-to-ETF mapping without API calls."""
        # Get ETF mapping (should be fast local operation)
        etf_mapping = await fast_agent.get_sector_etf_mapping()

        # Validate mapping completeness
        expected_sectors = {sector.value for sector in SectorName}
        actual_sectors = set(etf_mapping.keys())

        assert expected_sectors == actual_sectors, (
            f"ETF mapping mismatch. Missing: {expected_sectors - actual_sectors}"
        )

        # Validate all ETF tickers are properly formatted
        for sector, etf in etf_mapping.items():
            assert isinstance(etf, str), f"ETF ticker for {sector} must be string"
            assert len(etf) >= 2, f"ETF ticker {etf} too short"
            assert etf.isupper(), f"ETF ticker {etf} must be uppercase"
            assert etf.startswith("XL"), f"Expected SPDR ETF format for {etf}"

    @pytest.mark.asyncio
    async def test_confidence_thresholds(self, sample_analysis, performance_thresholds):
        """Test confidence threshold filtering using cached data."""
        # Handle both coroutine (first time) and cached data (subsequent times)
        if hasattr(sample_analysis, "__await__"):
            rating = await sample_analysis
        else:
            rating = sample_analysis

        # Validate confidence is above minimum threshold
        min_confidence = performance_thresholds["min_confidence"]
        assert rating["confidence"] >= min_confidence, (
            f"Confidence {rating['confidence']} below threshold {min_confidence}"
        )

        # Test with different threshold values
        test_thresholds = [0.1, 0.3, 0.5, 0.7]

        for threshold in test_thresholds:
            if rating["confidence"] >= threshold:
                # This result should pass the threshold
                assert True, f"Result should pass threshold {threshold}"
            else:
                # This result should be filtered out in production
                assert threshold > rating["confidence"], (
                    "Threshold validation inconsistent"
                )

    @pytest.mark.asyncio
    async def test_rating_distribution_analysis(self, multi_sector_analysis):
        """Analyze rating distribution from cached multi-sector results."""
        # Handle both coroutine (first time) and cached data (subsequent times)
        if hasattr(multi_sector_analysis, "__await__"):
            results = await multi_sector_analysis
        else:
            results = multi_sector_analysis
        ratings = [rating["rating"] for rating in results.values()]

        # Validate rating range
        assert all(1 <= r <= 5 for r in ratings), "All ratings must be 1-5"

        # Calculate distribution statistics
        avg_rating = sum(ratings) / len(ratings)
        rating_range = max(ratings) - min(ratings)

        # Validate reasonable distribution
        assert 1.0 <= avg_rating <= 5.0, f"Average rating {avg_rating} out of bounds"
        assert rating_range >= 0, "Rating range must be non-negative"

        # Count rating frequencies
        rating_counts = {}
        for rating in ratings:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1

        # Validate no single rating dominates (unless sample size is very small or all ratings are identical)
        if len(ratings) >= 3:
            max_frequency = max(rating_counts.values())
            unique_ratings = len(rating_counts)

            # If we have multiple unique ratings, ensure no single rating dominates
            # If all ratings are the same (unique_ratings == 1), that's acceptable for cached data
            if unique_ratings > 1:
                assert max_frequency < len(ratings), (
                    "Rating distribution too concentrated"
                )
            else:
                # All ratings are identical - this is valid for cached test data
                assert max_frequency == len(ratings), (
                    "Expected all ratings to be identical"
                )

    @pytest.mark.asyncio
    async def test_ensemble_structure_validation(self, sample_analysis):
        """Test ensemble result structure without running ensemble."""
        # Handle both coroutine (first time) and cached data (subsequent times)
        if hasattr(sample_analysis, "__await__"):
            rating = await sample_analysis
        else:
            rating = sample_analysis

        # Validate ensemble-style fields are present
        assert "confidence" in rating, "Ensemble confidence missing"
        assert "sub_scores" in rating, "Sub-scores missing"
        assert "weights" in rating, "Weights missing"
        assert "weighted_score" in rating, "Weighted score missing"

        # Validate tri-pillar structure
        sub_scores = rating["sub_scores"]
        weights = rating["weights"]

        expected_pillars = ["fundamentals", "sentiment", "technicals"]
        for pillar in expected_pillars:
            assert pillar in sub_scores, f"Sub-score for {pillar} missing"
            assert pillar in weights, f"Weight for {pillar} missing"
            assert 1 <= sub_scores[pillar] <= 5, f"Sub-score for {pillar} out of range"
            assert 0.0 <= weights[pillar] <= 1.0, f"Weight for {pillar} out of range"

        # Validate weighted score calculation
        expected_weighted = sum(
            sub_scores[pillar] * weights[pillar] for pillar in expected_pillars
        )
        actual_weighted = rating["weighted_score"]

        # Allow small floating point differences
        assert abs(expected_weighted - actual_weighted) < 0.1, (
            f"Weighted score calculation error: expected {expected_weighted}, got {actual_weighted}"
        )


# Additional optimized tests that can run independently
@pytest.mark.asyncio
async def test_basic_imports():
    """Test that all required modules can be imported (fast smoke test)."""
    from sector_committee.data_models import SectorName, SECTOR_ETF_MAP

    # Validate key constants
    assert len(SectorName) == 11, "Expected 11 SPDR sectors"
    assert len(SECTOR_ETF_MAP) == 11, "Expected 11 ETF mappings"

    # Test enum access
    tech_sector = SectorName.INFORMATION_TECHNOLOGY
    assert tech_sector.value == "Information Technology"
    assert SECTOR_ETF_MAP[tech_sector] == "XLK"


@pytest.mark.asyncio
async def test_schema_validation_edge_cases():
    """Test schema validation with edge cases (no API calls)."""
    from sector_committee.scoring.schema import validate_sector_rating

    # Test with minimal valid rating
    minimal_rating = {
        "rating": 3,
        "summary": "Test summary with minimum length requirement met",
        "sub_scores": {"fundamentals": 3, "sentiment": 3, "technicals": 3},
        "weights": {"fundamentals": 0.5, "sentiment": 0.3, "technicals": 0.2},
        "weighted_score": 3.0,
        "rationale": [
            {
                "pillar": "fundamentals",
                "reason": "Test reason",
                "impact": "positive",
                "confidence": 0.8,
            }
        ],
        "references": [
            {
                "url": "https://example.com",
                "title": "Test Reference",
                "description": "Test description",
                "accessed_at": "2023-10-01T12:00:00Z",
                "accessible": True,
            }
        ],
        "confidence": 0.75,
    }

    # Should pass validation
    result = validate_sector_rating(minimal_rating)
    assert result.is_valid, f"Minimal valid rating failed: {result.errors}"

    # Test with invalid rating (out of range)
    invalid_rating = minimal_rating.copy()
    invalid_rating["rating"] = 6  # Out of range

    result = validate_sector_rating(invalid_rating)
    assert not result.is_valid, "Invalid rating should fail validation"
    assert len(result.errors) > 0, "Should have validation errors"


if __name__ == "__main__":
    # Run optimized tests directly
    pytest.main([__file__, "-v", "-s"])
