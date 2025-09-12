"""Unit tests for schema validation module.

Fast tests with no external dependencies, focusing on JSON schema validation
logic, edge cases, and mathematical consistency checks.
"""

import pytest

from sector_committee.scoring.schema import (
    validate_sector_rating,
    ValidationResult,
    SECTOR_RATING_SCHEMA,
    fast_validate_sector_rating,
    validate_batch_ratings,
    create_openai_structured_output_schema,
    _validate_mathematical_consistency,
    _validate_business_logic,
    _map_weighted_to_rating,
)
from sector_committee.data_models import (
    SCORE_RANGE,
    CONFIDENCE_RANGE,
    WEIGHT_SUM_TOLERANCE,
)


class TestValidationResult:
    """Test ValidationResult class."""

    def test_validation_result_valid(self):
        """Test ValidationResult for valid case."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert bool(result) is True
        assert result.errors == []
        assert result.warnings == []
        assert str(result) == "Validation passed"

    def test_validation_result_invalid(self):
        """Test ValidationResult for invalid case."""
        errors = ["Missing field: rating", "Invalid range"]
        warnings = ["Low confidence"]
        result = ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        assert result.is_valid is False
        assert bool(result) is False
        assert result.errors == errors
        assert result.warnings == warnings
        assert "2 errors" in str(result)
        assert "1 warnings" in str(result)


class TestSectorRatingValidation:
    """Test sector rating validation functions."""

    @pytest.fixture
    def valid_rating(self):
        """Create a valid sector rating for testing."""
        return {
            "rating": 4,
            "summary": "Strong sector performance with positive fundamentals and growth prospects",
            "sub_scores": {"fundamentals": 4, "sentiment": 4, "technicals": 3},
            "weights": {"fundamentals": 0.5, "sentiment": 0.3, "technicals": 0.2},
            "weighted_score": 3.8,
            "rationale": [
                {
                    "pillar": "fundamentals",
                    "reason": "Strong earnings growth and solid balance sheets",
                    "impact": "positive",
                    "confidence": 0.85,
                },
                {
                    "pillar": "sentiment",
                    "reason": "Positive analyst sentiment and investor confidence",
                    "impact": "positive",
                    "confidence": 0.75,
                },
                {
                    "pillar": "technicals",
                    "reason": "Mixed technical indicators with some resistance levels",
                    "impact": "neutral",
                    "confidence": 0.60,
                },
            ],
            "references": [
                {
                    "url": "https://example.com/sector-analysis",
                    "title": "Sector Analysis Report",
                    "description": "Comprehensive analysis of sector fundamentals and outlook",
                    "accessed_at": "2025-01-01T12:00:00Z",
                    "accessible": True,
                }
            ],
            "confidence": 0.75,
        }

    def test_validate_valid_rating(self, valid_rating):
        """Test validation of valid sector rating."""
        result = validate_sector_rating(valid_rating)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_missing_required_field(self, valid_rating):
        """Test validation with missing required field."""
        invalid_rating = valid_rating.copy()
        del invalid_rating["rating"]

        result = validate_sector_rating(invalid_rating)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("required" in error.lower() for error in result.errors)

    def test_validate_out_of_range_rating(self, valid_rating):
        """Test validation with out-of-range rating."""
        invalid_rating = valid_rating.copy()
        invalid_rating["rating"] = 6  # Out of 1-5 range

        result = validate_sector_rating(invalid_rating)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_out_of_range_confidence(self, valid_rating):
        """Test validation with out-of-range confidence."""
        invalid_rating = valid_rating.copy()
        invalid_rating["confidence"] = 1.5  # Out of 0.0-1.0 range

        result = validate_sector_rating(invalid_rating)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_additional_properties_allowed(self, valid_rating):
        """Test that additional properties are now allowed."""
        rating_with_extra = valid_rating.copy()
        rating_with_extra["extra_field"] = "some_value"
        rating_with_extra["investment_thesis"] = "Strong fundamentals"

        result = validate_sector_rating(rating_with_extra)

        # Should pass with additional properties allowed
        assert result.is_valid is True


class TestMathematicalConsistency:
    """Test mathematical consistency validation."""

    def test_weight_sum_validation(self):
        """Test weight sum validation."""
        errors = []
        warnings = []

        # Valid weights
        data = {"weights": {"fundamentals": 0.5, "sentiment": 0.3, "technicals": 0.2}}
        _validate_mathematical_consistency(data, errors, warnings)
        assert len(errors) == 0

        # Invalid weights (sum > 1.0 + tolerance)
        errors.clear()
        data = {"weights": {"fundamentals": 0.6, "sentiment": 0.4, "technicals": 0.2}}
        _validate_mathematical_consistency(data, errors, warnings)
        assert len(errors) > 0
        assert any("sum to 1.0" in error for error in errors)

    def test_weighted_score_calculation(self):
        """Test weighted score calculation validation."""
        errors = []
        warnings = []

        # Valid calculation
        data = {
            "sub_scores": {"fundamentals": 4, "sentiment": 3, "technicals": 3},
            "weights": {"fundamentals": 0.5, "sentiment": 0.3, "technicals": 0.2},
            "weighted_score": 3.4,  # 4*0.5 + 3*0.3 + 3*0.2 = 3.4
        }
        _validate_mathematical_consistency(data, errors, warnings)
        assert len(errors) == 0

        # Invalid calculation (exceeds tolerance)
        errors.clear()
        data = {
            "sub_scores": {"fundamentals": 4, "sentiment": 3, "technicals": 3},
            "weights": {"fundamentals": 0.5, "sentiment": 0.3, "technicals": 0.2},
            "weighted_score": 2.0,  # Way off from 3.4
        }
        _validate_mathematical_consistency(data, errors, warnings)
        assert len(errors) > 0
        assert any("mismatch" in error.lower() for error in errors)

    def test_rating_consistency_warnings(self):
        """Test rating consistency warning generation."""
        errors = []
        warnings = []

        # Inconsistent rating should generate warning
        data = {
            "sub_scores": {"fundamentals": 5, "sentiment": 5, "technicals": 5},
            "weights": {"fundamentals": 0.5, "sentiment": 0.3, "technicals": 0.2},
            "weighted_score": 5.0,
            "rating": 2,  # Inconsistent with high weighted score
        }
        _validate_mathematical_consistency(data, errors, warnings)
        assert len(warnings) > 0


class TestBusinessLogicValidation:
    """Test business logic validation."""

    def test_rationale_coverage_validation(self):
        """Test rationale pillar coverage validation."""
        errors = []
        warnings = []

        # Complete coverage
        data = {
            "rationale": [
                {
                    "pillar": "fundamentals",
                    "reason": "test",
                    "impact": "positive",
                    "confidence": 0.8,
                },
                {
                    "pillar": "sentiment",
                    "reason": "test",
                    "impact": "positive",
                    "confidence": 0.8,
                },
                {
                    "pillar": "technicals",
                    "reason": "test",
                    "impact": "positive",
                    "confidence": 0.8,
                },
            ]
        }
        _validate_business_logic(data, errors, warnings)
        assert len(warnings) == 0  # Should have no missing pillar warnings

        # Incomplete coverage
        warnings.clear()
        data = {
            "rationale": [
                {
                    "pillar": "fundamentals",
                    "reason": "test",
                    "impact": "positive",
                    "confidence": 0.8,
                }
            ]
        }
        _validate_business_logic(data, errors, warnings)
        assert len(warnings) > 0
        assert any("missing rationale" in warning.lower() for warning in warnings)

    def test_reference_accessibility_validation(self):
        """Test reference accessibility validation."""
        errors = []
        warnings = []

        # All accessible references
        data = {
            "references": [
                {"accessible": True, "url": "test1"},
                {"accessible": True, "url": "test2"},
            ]
        }
        _validate_business_logic(data, errors, warnings)
        # Should not add accessibility warnings

        # No accessible references
        warnings.clear()
        data = {
            "references": [
                {"accessible": False, "url": "test1"},
                {"accessible": False, "url": "test2"},
            ]
        }
        _validate_business_logic(data, errors, warnings)
        assert len(warnings) > 0
        assert any(
            "no references are accessible" in warning.lower() for warning in warnings
        )

    def test_confidence_calibration_validation(self):
        """Test confidence calibration warnings."""
        errors = []
        warnings = []

        # High rating with low confidence
        data = {"rating": 5, "confidence": 0.4}
        _validate_business_logic(data, errors, warnings)
        assert len(warnings) > 0
        assert any(
            "high rating" in warning.lower() and "low confidence" in warning.lower()
            for warning in warnings
        )

        # Low rating with high confidence
        warnings.clear()
        data = {"rating": 1, "confidence": 0.9}
        _validate_business_logic(data, errors, warnings)
        assert len(warnings) > 0
        assert any(
            "low rating" in warning.lower() and "high confidence" in warning.lower()
            for warning in warnings
        )


class TestUtilityFunctions:
    """Test utility functions."""

    def test_map_weighted_to_rating(self):
        """Test weighted score to rating mapping."""
        # Test all rating ranges
        assert _map_weighted_to_rating(1.0) == 1
        assert _map_weighted_to_rating(1.4) == 1
        assert _map_weighted_to_rating(1.5) == 2
        assert _map_weighted_to_rating(2.4) == 2
        assert _map_weighted_to_rating(2.5) == 3
        assert _map_weighted_to_rating(3.4) == 3
        assert _map_weighted_to_rating(3.5) == 4
        assert _map_weighted_to_rating(4.4) == 4
        assert _map_weighted_to_rating(4.5) == 5
        assert _map_weighted_to_rating(5.0) == 5

    def test_fast_validate_sector_rating(self):
        """Test fast validation function."""
        # Valid rating
        valid_rating = {
            "rating": 3,
            "summary": "Test summary that meets minimum length requirements",
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
                    "description": "Test description that meets minimum length",
                    "accessed_at": "2025-01-01T12:00:00Z",
                    "accessible": True,
                }
            ],
            "confidence": 0.75,
        }

        assert fast_validate_sector_rating(valid_rating) is True

        # Invalid rating (missing required field)
        invalid_rating = valid_rating.copy()
        del invalid_rating["rating"]

        assert fast_validate_sector_rating(invalid_rating) is False


class TestSchemaStructure:
    """Test schema structure and consistency."""

    def test_schema_required_fields(self):
        """Test that schema has all required fields."""
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
            assert field in SECTOR_RATING_SCHEMA["properties"]
            assert field in SECTOR_RATING_SCHEMA["required"]

    def test_schema_score_ranges(self):
        """Test that schema uses correct score ranges."""
        rating_props = SECTOR_RATING_SCHEMA["properties"]["rating"]
        assert rating_props["minimum"] == SCORE_RANGE[0]
        assert rating_props["maximum"] == SCORE_RANGE[1]

        confidence_props = SECTOR_RATING_SCHEMA["properties"]["confidence"]
        assert confidence_props["minimum"] == CONFIDENCE_RANGE[0]
        assert confidence_props["maximum"] == CONFIDENCE_RANGE[1]

    def test_schema_sub_scores_structure(self):
        """Test sub_scores schema structure."""
        sub_scores_props = SECTOR_RATING_SCHEMA["properties"]["sub_scores"]
        expected_pillars = ["fundamentals", "sentiment", "technicals"]

        for pillar in expected_pillars:
            assert pillar in sub_scores_props["properties"]
            assert pillar in sub_scores_props["required"]

            pillar_props = sub_scores_props["properties"][pillar]
            assert pillar_props["minimum"] == SCORE_RANGE[0]
            assert pillar_props["maximum"] == SCORE_RANGE[1]

    def test_schema_weights_structure(self):
        """Test weights schema structure."""
        weights_props = SECTOR_RATING_SCHEMA["properties"]["weights"]
        expected_pillars = ["fundamentals", "sentiment", "technicals"]

        for pillar in expected_pillars:
            assert pillar in weights_props["properties"]
            assert pillar in weights_props["required"]

            pillar_props = weights_props["properties"][pillar]
            assert pillar_props["minimum"] == 0.0
            assert pillar_props["maximum"] == 1.0

    def test_openai_structured_output_schema(self):
        """Test OpenAI structured output schema generation."""
        schema = create_openai_structured_output_schema()

        # Should have same structure as main schema
        assert "properties" in schema
        assert "required" in schema

        # Should have all required fields
        for field in SECTOR_RATING_SCHEMA["required"]:
            assert field in schema["required"]
            assert field in schema["properties"]


class TestBatchValidation:
    """Test batch validation functionality."""

    def test_validate_batch_ratings_all_valid(self):
        """Test batch validation with all valid ratings."""
        valid_rating = {
            "rating": 3,
            "summary": "Test summary that meets minimum length requirements",
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
                    "description": "Test description that meets minimum length",
                    "accessed_at": "2025-01-01T12:00:00Z",
                    "accessible": True,
                }
            ],
            "confidence": 0.75,
        }

        ratings = [valid_rating, valid_rating.copy(), valid_rating.copy()]
        results, valid_count = validate_batch_ratings(ratings)

        assert len(results) == 3
        assert valid_count == 3
        assert all(result.is_valid for result in results)

    def test_validate_batch_ratings_mixed(self):
        """Test batch validation with mixed valid/invalid ratings."""
        valid_rating = {
            "rating": 3,
            "summary": "Test summary that meets minimum length requirements",
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
                    "description": "Test description that meets minimum length",
                    "accessed_at": "2025-01-01T12:00:00Z",
                    "accessible": True,
                }
            ],
            "confidence": 0.75,
        }

        invalid_rating = valid_rating.copy()
        del invalid_rating["rating"]  # Missing required field

        ratings = [valid_rating, invalid_rating, valid_rating.copy()]
        results, valid_count = validate_batch_ratings(ratings)

        assert len(results) == 3
        assert valid_count == 2
        assert results[0].is_valid is True
        assert results[1].is_valid is False
        assert results[2].is_valid is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_rationale_array(self):
        """Test validation with empty rationale array."""
        rating = {
            "rating": 3,
            "summary": "Test summary with minimum length requirement met",
            "sub_scores": {"fundamentals": 3, "sentiment": 3, "technicals": 3},
            "weights": {"fundamentals": 0.5, "sentiment": 0.3, "technicals": 0.2},
            "weighted_score": 3.0,
            "rationale": [],  # Empty array
            "references": [
                {
                    "url": "https://example.com",
                    "title": "Test Reference",
                    "description": "Test description that meets minimum length",
                    "accessed_at": "2025-01-01T12:00:00Z",
                    "accessible": True,
                }
            ],
            "confidence": 0.75,
        }

        result = validate_sector_rating(rating)
        assert result.is_valid is False  # Should fail due to minItems: 1

    def test_empty_references_array(self):
        """Test validation with empty references array."""
        rating = {
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
            "references": [],  # Empty array
            "confidence": 0.75,
        }

        result = validate_sector_rating(rating)
        assert result.is_valid is False  # Should fail due to minItems: 1

    def test_boundary_values(self):
        """Test boundary values for numeric fields."""
        base_rating = {
            "rating": 1,  # Minimum score
            "summary": "A" * 10,  # Minimum length
            "sub_scores": {"fundamentals": 1, "sentiment": 1, "technicals": 1},
            "weights": {"fundamentals": 0.5, "sentiment": 0.3, "technicals": 0.2},
            "weighted_score": 1.0,
            "rationale": [
                {
                    "pillar": "fundamentals",
                    "reason": "A" * 5,  # Minimum length
                    "impact": "negative",
                    "confidence": 0.0,  # Minimum confidence
                }
            ],
            "references": [
                {
                    "url": "https://ex.co",  # Minimum length (10 chars)
                    "title": "A" * 5,  # Minimum length
                    "description": "A" * 10,  # Minimum length
                    "accessed_at": "2025-01-01T12:00:00Z",
                    "accessible": False,
                }
            ],
            "confidence": 0.0,  # Minimum confidence
        }

        result = validate_sector_rating(base_rating)
        assert result.is_valid is True

    def test_maximum_boundary_values(self):
        """Test maximum boundary values."""
        base_rating = {
            "rating": 5,  # Maximum score
            "summary": "A" * 500,  # Maximum length
            "sub_scores": {"fundamentals": 5, "sentiment": 5, "technicals": 5},
            "weights": {"fundamentals": 0.5, "sentiment": 0.3, "technicals": 0.2},
            "weighted_score": 5.0,
            "rationale": [
                {
                    "pillar": "fundamentals",
                    "reason": "A" * 200,  # Maximum length
                    "impact": "positive",
                    "confidence": 1.0,  # Maximum confidence
                }
            ],
            "references": [
                {
                    "url": "https://example.com",
                    "title": "A" * 150,  # Maximum length
                    "description": "A" * 300,  # Maximum length
                    "accessed_at": "2025-01-01T12:00:00Z",
                    "accessible": True,
                }
            ],
            "confidence": 1.0,  # Maximum confidence
        }

        result = validate_sector_rating(base_rating)
        assert result.is_valid is True


class TestConstants:
    """Test module constants."""

    def test_score_range_usage(self):
        """Test that SCORE_RANGE is used consistently."""
        assert SCORE_RANGE[0] == 1
        assert SCORE_RANGE[1] == 5

        # Check usage in schema
        rating_props = SECTOR_RATING_SCHEMA["properties"]["rating"]
        assert rating_props["minimum"] == SCORE_RANGE[0]
        assert rating_props["maximum"] == SCORE_RANGE[1]

    def test_confidence_range_usage(self):
        """Test that CONFIDENCE_RANGE is used consistently."""
        assert CONFIDENCE_RANGE[0] == 0.0
        assert CONFIDENCE_RANGE[1] == 1.0

        # Check usage in schema
        confidence_props = SECTOR_RATING_SCHEMA["properties"]["confidence"]
        assert confidence_props["minimum"] == CONFIDENCE_RANGE[0]
        assert confidence_props["maximum"] == CONFIDENCE_RANGE[1]

    def test_weight_sum_tolerance_usage(self):
        """Test that WEIGHT_SUM_TOLERANCE is reasonable."""
        assert WEIGHT_SUM_TOLERANCE == 0.01
        assert 0.001 <= WEIGHT_SUM_TOLERANCE <= 0.1


if __name__ == "__main__":
    # Run unit tests directly
    pytest.main([__file__, "-v"])
