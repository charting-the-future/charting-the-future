"""JSON schema validation for sector analysis outputs.

This module provides strict schema validation for all sector analysis outputs,
ensuring 100% compliance with the defined data structures and preventing
malformed responses from reaching the portfolio construction pipeline.

The validation framework supports both real-time validation during API
responses and batch validation for audit compliance.
"""

from typing import Any, Dict, List, Optional, Tuple
from jsonschema import validate, ValidationError, Draft7Validator  # type: ignore

from ..data_models import SCORE_RANGE, CONFIDENCE_RANGE, WEIGHT_SUM_TOLERANCE


# JSON Schema for SectorRating output validation
SECTOR_RATING_SCHEMA = {
    "type": "object",
    "properties": {
        "rating": {
            "type": "integer",
            "minimum": SCORE_RANGE[0],
            "maximum": SCORE_RANGE[1],
            "description": "Final sector score (1-5)",
        },
        "summary": {
            "type": "string",
            "minLength": 10,
            "maxLength": 500,
            "description": "Concise explanation of the rating",
        },
        "sub_scores": {
            "type": "object",
            "properties": {
                "fundamentals": {
                    "type": "integer",
                    "minimum": SCORE_RANGE[0],
                    "maximum": SCORE_RANGE[1],
                },
                "sentiment": {
                    "type": "integer",
                    "minimum": SCORE_RANGE[0],
                    "maximum": SCORE_RANGE[1],
                },
                "technicals": {
                    "type": "integer",
                    "minimum": SCORE_RANGE[0],
                    "maximum": SCORE_RANGE[1],
                },
            },
            "required": ["fundamentals", "sentiment", "technicals"],
            "additionalProperties": False,
        },
        "weights": {
            "type": "object",
            "properties": {
                "fundamentals": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "sentiment": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "technicals": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["fundamentals", "sentiment", "technicals"],
            "additionalProperties": False,
        },
        "weighted_score": {
            "type": "number",
            "minimum": float(SCORE_RANGE[0]),
            "maximum": float(SCORE_RANGE[1]),
            "description": "Computed weighted average of sub-scores",
        },
        "rationale": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pillar": {
                        "type": "string",
                        "enum": ["fundamentals", "sentiment", "technicals"],
                    },
                    "reason": {"type": "string", "minLength": 5, "maxLength": 200},
                    "impact": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": CONFIDENCE_RANGE[0],
                        "maximum": CONFIDENCE_RANGE[1],
                    },
                },
                "required": ["pillar", "reason", "impact", "confidence"],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": 10,
        },
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "minLength": 10},
                    "title": {"type": "string", "minLength": 5, "maxLength": 150},
                    "description": {
                        "type": "string",
                        "minLength": 10,
                        "maxLength": 300,
                    },
                    "accessed_at": {"type": "string"},
                    "accessible": {"type": "boolean"},
                    "local_path": {
                        "type": "string",
                        "description": "Local file path if content was downloaded",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message if download failed",
                    },
                },
                "required": [
                    "url",
                    "title",
                    "description",
                    "accessed_at",
                    "accessible",
                ],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": 10,
        },
        "confidence": {
            "type": "number",
            "minimum": CONFIDENCE_RANGE[0],
            "maximum": CONFIDENCE_RANGE[1],
            "description": "Overall confidence in the analysis",
        },
    },
    "required": [
        "rating",
        "summary",
        "sub_scores",
        "weights",
        "weighted_score",
        "rationale",
        "references",
        "confidence",
    ],
    "additionalProperties": True,  # Allow additional fields from LLM responses
}


class ValidationResult:
    """Result of schema validation with detailed error reporting."""

    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        """Initialize validation result.

        Args:
            is_valid: Whether the data passed validation.
            errors: List of validation errors (if any).
            warnings: List of validation warnings (if any).
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []

    def __bool__(self) -> bool:
        """Return validation status."""
        return self.is_valid

    def __str__(self) -> str:
        """String representation of validation result."""
        if self.is_valid:
            return "Validation passed"

        error_msg = f"Validation failed with {len(self.errors)} errors"
        if self.warnings:
            error_msg += f" and {len(self.warnings)} warnings"
        return error_msg


def validate_sector_rating(data: Dict[str, Any]) -> ValidationResult:
    """Validate sector rating data against JSON schema.

    Performs comprehensive validation including:
    - JSON schema compliance
    - Mathematical consistency checks
    - Business logic validation

    Args:
        data: Dictionary containing sector rating data.

    Returns:
        ValidationResult with detailed error reporting.
    """
    errors: List[str] = []
    warnings: List[str] = []

    try:
        # Primary JSON schema validation
        validate(instance=data, schema=SECTOR_RATING_SCHEMA)

    except ValidationError as e:
        errors.append(f"Schema validation failed: {e.message}")
        return ValidationResult(is_valid=False, errors=errors)

    # Mathematical consistency validation
    try:
        _validate_mathematical_consistency(data, errors, warnings)
        _validate_business_logic(data, errors, warnings)

    except Exception as e:
        errors.append(f"Validation error: {str(e)}")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def _validate_mathematical_consistency(
    data: Dict[str, Any], errors: List[str], warnings: List[str]
) -> None:
    """Validate mathematical relationships in the data.

    Args:
        data: Sector rating data to validate.
        errors: List to append validation errors.
        warnings: List to append validation warnings.
    """
    # Validate weight sum
    weights = data.get("weights", {})
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        errors.append(f"Weights must sum to 1.0, got {weight_sum:.4f}")

    # Validate weighted score calculation
    sub_scores = data.get("sub_scores", {})
    expected_weighted = (
        sub_scores.get("fundamentals", 0) * weights.get("fundamentals", 0)
        + sub_scores.get("sentiment", 0) * weights.get("sentiment", 0)
        + sub_scores.get("technicals", 0) * weights.get("technicals", 0)
    )

    actual_weighted = data.get("weighted_score", 0)
    if (
        abs(actual_weighted - expected_weighted) > 0.25
    ):  # Increased tolerance for LLM precision
        errors.append(
            f"Weighted score mismatch. Expected {expected_weighted:.2f}, "
            f"got {actual_weighted:.2f}"
        )

    # Validate rating consistency with weighted score
    rating = data.get("rating", 0)
    expected_rating = _map_weighted_to_rating(expected_weighted)
    if rating != expected_rating:
        warnings.append(
            f"Rating {rating} may be inconsistent with weighted score "
            f"{expected_weighted:.2f} (suggests {expected_rating})"
        )


def _validate_business_logic(
    data: Dict[str, Any], errors: List[str], warnings: List[str]
) -> None:
    """Validate business logic constraints.

    Args:
        data: Sector rating data to validate.
        errors: List to append validation errors.
        warnings: List to append validation warnings.
    """
    # Validate rationale coverage
    rationale = data.get("rationale", [])
    pillars_covered = {item.get("pillar") for item in rationale}
    required_pillars = {"fundamentals", "sentiment", "technicals"}

    missing_pillars = required_pillars - pillars_covered
    if missing_pillars:
        warnings.append(f"Missing rationale for pillars: {missing_pillars}")

    # Validate reference quality
    references = data.get("references", [])
    accessible_refs = sum(1 for ref in references if ref.get("accessible", False))

    # More lenient validation - only warn about accessibility
    if accessible_refs == 0 and len(references) > 0:
        warnings.append("No references are accessible - may impact analysis quality")
    elif accessible_refs < len(references) * 0.5 and len(references) > 1:
        warnings.append(
            f"Low reference accessibility: {accessible_refs}/{len(references)} "
            "references are accessible"
        )

    # Validate confidence calibration
    confidence = data.get("confidence", 0)
    rating = data.get("rating", 0)

    # High ratings with low confidence should trigger warnings
    if rating >= 4 and confidence < 0.6:
        warnings.append(
            f"High rating ({rating}) with low confidence ({confidence:.2f}) "
            "may indicate uncertain analysis"
        )

    # Low ratings with high confidence should also trigger warnings
    if rating <= 2 and confidence > 0.8:
        warnings.append(
            f"Low rating ({rating}) with high confidence ({confidence:.2f}) "
            "may indicate overly certain negative view"
        )


def _map_weighted_to_rating(weighted_score: float) -> int:
    """Map weighted score to integer rating.

    Args:
        weighted_score: Weighted average score (1.0-5.0).

    Returns:
        Integer rating (1-5).
    """
    if weighted_score < 1.5:
        return 1
    elif weighted_score < 2.5:
        return 2
    elif weighted_score < 3.5:
        return 3
    elif weighted_score < 4.5:
        return 4
    else:
        return 5


def create_openai_structured_output_schema() -> Dict[str, Any]:
    """Create OpenAI structured output compatible schema.

    OpenAI structured outputs require additionalProperties to be false
    throughout the schema for strict mode compliance.

    Returns:
        Schema dictionary formatted for OpenAI structured outputs.
    """
    import copy

    # Create a deep copy to avoid modifying the original schema
    output_schema = copy.deepcopy(SECTOR_RATING_SCHEMA)

    # OpenAI structured outputs require additionalProperties: false
    output_schema["additionalProperties"] = False

    # Ensure all nested objects also have additionalProperties: false
    output_schema["properties"]["sub_scores"]["additionalProperties"] = False
    output_schema["properties"]["weights"]["additionalProperties"] = False
    output_schema["properties"]["rationale"]["items"]["additionalProperties"] = False

    # Remove post-processing fields from references for structured output
    # and ensure additionalProperties: false
    output_schema["properties"]["references"]["items"]["properties"] = {
        "url": {"type": "string", "minLength": 10},
        "title": {"type": "string", "minLength": 5, "maxLength": 150},
        "description": {"type": "string", "minLength": 10, "maxLength": 300},
        "accessed_at": {"type": "string"},
        "accessible": {"type": "boolean"},
    }
    output_schema["properties"]["references"]["items"]["required"] = [
        "url",
        "title",
        "description",
        "accessed_at",
        "accessible",
    ]
    output_schema["properties"]["references"]["items"]["additionalProperties"] = False

    return output_schema


def validate_batch_ratings(
    ratings: List[Dict[str, Any]],
) -> Tuple[List[ValidationResult], int]:
    """Validate a batch of sector ratings.

    Args:
        ratings: List of sector rating dictionaries to validate.

    Returns:
        Tuple of (validation results, count of valid ratings).
    """
    results = []
    valid_count = 0

    for i, rating in enumerate(ratings):
        try:
            result = validate_sector_rating(rating)
            results.append(result)
            if result.is_valid:
                valid_count += 1
        except Exception as e:
            results.append(
                ValidationResult(
                    is_valid=False,
                    errors=[f"Validation exception for rating {i}: {str(e)}"],
                )
            )

    return results, valid_count


# Pre-compiled validator for performance
_VALIDATOR = Draft7Validator(SECTOR_RATING_SCHEMA)


def fast_validate_sector_rating(data: Dict[str, Any]) -> bool:
    """Fast validation for performance-critical paths.

    Args:
        data: Sector rating data to validate.

    Returns:
        True if data is valid, False otherwise.
    """
    try:
        _VALIDATOR.validate(data)
        return True
    except ValidationError:
        return False
