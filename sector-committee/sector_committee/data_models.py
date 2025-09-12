"""Data models for sector analysis system.

This module defines the core data structures used throughout the sector analysis
pipeline, including request specifications and response formats with strict
typing and validation.

All models follow TypedDict patterns for JSON schema compatibility and use
comprehensive type hints for development safety.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict, TYPE_CHECKING
from enum import Enum

from dotenv import load_dotenv

# Import ModelName from llm_models (avoid circular import)
if TYPE_CHECKING:
    from .llm_models import ModelName

load_dotenv(override=True)


class SectorName(Enum):
    """11 SPDR sector ETF mappings for systematic analysis."""

    COMMUNICATION_SERVICES = "Communication Services"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    ENERGY = "Energy"
    FINANCIALS = "Financials"
    HEALTH_CARE = "Health Care"
    INDUSTRIALS = "Industrials"
    INFORMATION_TECHNOLOGY = "Information Technology"
    MATERIALS = "Materials"
    REAL_ESTATE = "Real Estate"
    UTILITIES = "Utilities"


# SPDR sector ETF mappings for validation
SECTOR_ETF_MAP = {
    SectorName.COMMUNICATION_SERVICES: "XLC",
    SectorName.CONSUMER_DISCRETIONARY: "XLY",
    SectorName.CONSUMER_STAPLES: "XLP",
    SectorName.ENERGY: "XLE",
    SectorName.FINANCIALS: "XLF",
    SectorName.HEALTH_CARE: "XLV",
    SectorName.INDUSTRIALS: "XLI",
    SectorName.INFORMATION_TECHNOLOGY: "XLK",
    SectorName.MATERIALS: "XLB",
    SectorName.REAL_ESTATE: "XLRE",
    SectorName.UTILITIES: "XLU",
}


@dataclass(frozen=True)
class SectorRequest:
    """Request specification for sector analysis.

    Attributes:
        sector: Name of the sector to analyze (must be valid SPDR sector).
        horizon_weeks: Investment horizon in weeks (default: 4).
        weights_hint: Optional pillar weight overrides for analysis.
    """

    sector: str
    horizon_weeks: int = 4
    weights_hint: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        """Validate request parameters."""
        if self.horizon_weeks < 1 or self.horizon_weeks > 52:
            raise ValueError(
                f"Invalid horizon_weeks: {self.horizon_weeks}. Must be 1-52."
            )

        if self.weights_hint:
            weight_sum = sum(self.weights_hint.values())
            if abs(weight_sum - 1.0) > 0.01:
                raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")


class SubScores(TypedDict):
    """Tri-pillar sub-scores for sector analysis."""

    fundamentals: int  # 1-5 range
    sentiment: int  # 1-5 range
    technicals: int  # 1-5 range


class PillarWeights(TypedDict):
    """Weights for combining tri-pillar scores."""

    fundamentals: float  # 0-1 range, must sum to 1.0
    sentiment: float  # 0-1 range, must sum to 1.0
    technicals: float  # 0-1 range, must sum to 1.0


class RationaleItem(TypedDict):
    """Supporting evidence for sector analysis."""

    pillar: str  # "fundamentals", "sentiment", or "technicals"
    reason: str  # Explanation of the evidence
    impact: str  # "positive", "negative", or "neutral"
    confidence: float  # 0-1 confidence in this evidence


class ReferenceItem(TypedDict):
    """Source reference with validation metadata."""

    url: str  # Source URL
    title: str  # Article/document title
    description: str  # Brief content summary
    accessed_at: str  # UTC timestamp of access
    accessible: bool  # Whether URL was successfully retrieved


class SectorRating(TypedDict):
    """Complete sector analysis output with validation.

    This is the primary output format for the sector analysis system.
    All fields are required and must pass JSON schema validation.
    """

    rating: int  # Final 1-5 score (REQUIRED)
    summary: str  # Concise explanation (10-500 chars)
    sub_scores: SubScores  # Tri-pillar breakdown
    weights: PillarWeights  # Applied pillar weights
    weighted_score: float  # Mathematical roll-up (1.0-5.0)
    rationale: List[RationaleItem]  # Supporting evidence chain
    references: List[ReferenceItem]  # Cited sources with metadata
    confidence: float  # Overall reliability (0-1)


@dataclass
class ModelResult:
    """Result container for individual model responses."""

    model: "ModelName"
    data: SectorRating
    latency_ms: float
    timestamp_utc: str
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class EnsembleResult:
    """Result container for ensemble analysis."""

    final_rating: SectorRating
    model_results: List[ModelResult]
    consensus_score: float  # Agreement level between models (0-1)
    total_latency_ms: float
    timestamp_utc: str


# Default pillar weights for sector analysis
DEFAULT_WEIGHTS: PillarWeights = {
    "fundamentals": 0.5,  # 50% weight on fundamental analysis
    "sentiment": 0.3,  # 30% weight on market sentiment
    "technicals": 0.2,  # 20% weight on technical indicators
}

# Validation ranges for scores and weights
SCORE_RANGE = (1, 5)  # Valid range for all scores
CONFIDENCE_RANGE = (0.0, 1.0)  # Valid range for confidence values
WEIGHT_SUM_TOLERANCE = 0.01  # Tolerance for weight sum validation
