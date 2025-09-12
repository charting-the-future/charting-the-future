"""Unit tests for data models module.

Fast tests with no external dependencies, focusing on data model validation,
enum consistency, and dataclass behavior.
"""

import pytest
from datetime import datetime, timezone

from sector_committee.data_models import (
    SectorName,
    SectorRequest,
    ModelResult,
    SECTOR_ETF_MAP,
    SCORE_RANGE,
    CONFIDENCE_RANGE,
    WEIGHT_SUM_TOLERANCE,
)
from sector_committee.llm_models import ModelName


class TestModelName:
    """Test ModelName enum."""

    def test_openai_models(self):
        """Test OpenAI model names."""
        assert ModelName.OPENAI_GPT4.value == "gpt-4"
        assert ModelName.OPENAI_GPT4O.value == "gpt-4o"
        assert ModelName.OPENAI_GPT4O_STRUCTURED.value == "gpt-4o-2024-08-06"
        assert ModelName.OPENAI_O3_DEEP_RESEARCH.value == "o3-deep-research"
        assert ModelName.OPENAI_O4_MINI_DEEP_RESEARCH.value == "o4-mini-deep-research"

    def test_claude_models(self):
        """Test Claude model names."""
        assert ModelName.CLAUDE_SONNET.value == "claude-3-sonnet"
        assert ModelName.CLAUDE_HAIKU.value == "claude-3-haiku"

    def test_gemini_models(self):
        """Test Gemini model names."""
        assert ModelName.GEMINI_PRO.value == "gemini-pro"
        assert ModelName.GEMINI_FLASH.value == "gemini-1.5-flash"

    def test_model_name_uniqueness(self):
        """Test that all model names are unique."""
        values = [model.value for model in ModelName]
        assert len(values) == len(set(values)), "Model names must be unique"

    def test_model_name_count(self):
        """Test expected number of models."""
        # Verify we have the expected number of models across providers
        assert len(ModelName) >= 9, "Should have at least 9 model variants"


class TestSectorName:
    """Test SectorName enum."""

    def test_sector_names(self):
        """Test all 11 SPDR sector names."""
        expected_sectors = [
            "Information Technology",
            "Health Care",
            "Financials",
            "Communication Services",
            "Consumer Discretionary",
            "Industrials",
            "Consumer Staples",
            "Energy",
            "Utilities",
            "Real Estate",
            "Materials",
        ]

        actual_sectors = [sector.value for sector in SectorName]
        assert len(actual_sectors) == 11, "Should have exactly 11 SPDR sectors"

        for expected in expected_sectors:
            assert expected in actual_sectors, f"Missing sector: {expected}"

    def test_sector_etf_mapping_completeness(self):
        """Test that ETF mapping covers all sectors."""
        assert len(SECTOR_ETF_MAP) == len(SectorName), (
            "ETF mapping should cover all sectors"
        )

        for sector in SectorName:
            assert sector in SECTOR_ETF_MAP, f"Missing ETF mapping for {sector.value}"
            etf = SECTOR_ETF_MAP[sector]
            assert isinstance(etf, str), f"ETF for {sector.value} should be string"
            assert len(etf) >= 2, f"ETF {etf} too short"
            assert etf.isupper(), f"ETF {etf} should be uppercase"

    def test_etf_mapping_values(self):
        """Test specific ETF mappings."""
        assert SECTOR_ETF_MAP[SectorName.INFORMATION_TECHNOLOGY] == "XLK"
        assert SECTOR_ETF_MAP[SectorName.HEALTH_CARE] == "XLV"
        assert SECTOR_ETF_MAP[SectorName.FINANCIALS] == "XLF"
        assert SECTOR_ETF_MAP[SectorName.COMMUNICATION_SERVICES] == "XLC"
        assert SECTOR_ETF_MAP[SectorName.CONSUMER_DISCRETIONARY] == "XLY"
        assert SECTOR_ETF_MAP[SectorName.INDUSTRIALS] == "XLI"
        assert SECTOR_ETF_MAP[SectorName.CONSUMER_STAPLES] == "XLP"
        assert SECTOR_ETF_MAP[SectorName.ENERGY] == "XLE"
        assert SECTOR_ETF_MAP[SectorName.UTILITIES] == "XLU"
        assert SECTOR_ETF_MAP[SectorName.REAL_ESTATE] == "XLRE"
        assert SECTOR_ETF_MAP[SectorName.MATERIALS] == "XLB"


class TestSectorRequest:
    """Test SectorRequest dataclass."""

    def test_sector_request_creation(self):
        """Test basic SectorRequest creation."""
        request = SectorRequest(sector="Information Technology", horizon_weeks=4)

        assert request.sector == "Information Technology"
        assert request.horizon_weeks == 4
        assert request.weights_hint is None

    def test_sector_request_with_weights(self):
        """Test SectorRequest with custom weights."""
        weights = {"fundamentals": 0.5, "sentiment": 0.3, "technicals": 0.2}
        request = SectorRequest(
            sector="Health Care", horizon_weeks=8, weights_hint=weights
        )

        assert request.sector == "Health Care"
        assert request.horizon_weeks == 8
        assert request.weights_hint == weights

    def test_sector_request_validation(self):
        """Test SectorRequest parameter validation."""
        # Valid cases
        valid_sectors = ["Information Technology", "Health Care", "Financials"]
        for sector in valid_sectors:
            request = SectorRequest(sector=sector, horizon_weeks=4)
            assert request.sector == sector

        # Valid horizon weeks (1-52)
        for weeks in [1, 4, 13, 26, 52]:
            request = SectorRequest(sector="Utilities", horizon_weeks=weeks)
            assert request.horizon_weeks == weeks

    def test_sector_request_equality(self):
        """Test SectorRequest equality comparison."""
        request1 = SectorRequest(sector="Energy", horizon_weeks=4)
        request2 = SectorRequest(sector="Energy", horizon_weeks=4)
        request3 = SectorRequest(sector="Energy", horizon_weeks=8)

        assert request1 == request2
        assert request1 != request3


class TestModelResult:
    """Test ModelResult dataclass."""

    def test_model_result_creation(self):
        """Test basic ModelResult creation."""
        test_data = {
            "rating": 4,
            "summary": "Test analysis summary",
            "confidence": 0.85,
        }

        result = ModelResult(
            model=ModelName.OPENAI_GPT4,
            data=test_data,
            latency_ms=1250.5,
            timestamp_utc="2025-01-01T12:00:00Z",
            success=True,
        )

        assert result.model == ModelName.OPENAI_GPT4
        assert result.data == test_data
        assert result.latency_ms == 1250.5
        assert result.timestamp_utc == "2025-01-01T12:00:00Z"
        assert result.success is True

    def test_model_result_failure(self):
        """Test ModelResult for failed analysis."""
        result = ModelResult(
            model=ModelName.OPENAI_GPT4O,
            data={},
            latency_ms=500.0,
            timestamp_utc="2025-01-01T12:00:00Z",
            success=False,
        )

        assert result.success is False
        assert result.data == {}

    def test_model_result_timestamp_format(self):
        """Test timestamp format validation."""
        # Should accept ISO format
        iso_time = datetime.now(timezone.utc).isoformat()
        result = ModelResult(
            model=ModelName.OPENAI_GPT4,
            data={"rating": 3},
            latency_ms=1000.0,
            timestamp_utc=iso_time,
            success=True,
        )
        assert result.timestamp_utc == iso_time


class TestConstants:
    """Test module constants."""

    def test_score_range(self):
        """Test SCORE_RANGE constant."""
        assert SCORE_RANGE == (1, 5)
        assert isinstance(SCORE_RANGE[0], int)
        assert isinstance(SCORE_RANGE[1], int)
        assert SCORE_RANGE[0] < SCORE_RANGE[1]

    def test_confidence_range(self):
        """Test CONFIDENCE_RANGE constant."""
        assert CONFIDENCE_RANGE == (0.0, 1.0)
        assert isinstance(CONFIDENCE_RANGE[0], float)
        assert isinstance(CONFIDENCE_RANGE[1], float)
        assert CONFIDENCE_RANGE[0] < CONFIDENCE_RANGE[1]

    def test_weight_sum_tolerance(self):
        """Test WEIGHT_SUM_TOLERANCE constant."""
        assert WEIGHT_SUM_TOLERANCE == 0.01
        assert isinstance(WEIGHT_SUM_TOLERANCE, float)
        assert WEIGHT_SUM_TOLERANCE > 0


class TestEnumIntegration:
    """Test integration between different enums and constants."""

    def test_sector_name_etf_coverage(self):
        """Test that ETF mapping covers all sectors comprehensively."""
        # All sectors should have ETF mappings
        for sector in SectorName:
            assert sector in SECTOR_ETF_MAP
            etf = SECTOR_ETF_MAP[sector]

            # All ETFs should follow SPDR pattern (mostly XL*)
            assert etf.startswith("XL") or etf == "XLRE", (
                f"Unexpected ETF format: {etf}"
            )

    def test_model_name_provider_coverage(self):
        """Test that models cover major providers."""
        model_values = [model.value for model in ModelName]

        # Should have OpenAI models
        openai_models = [
            m for m in model_values if m.startswith("gpt") or m.startswith("o")
        ]
        assert len(openai_models) >= 5, "Should have multiple OpenAI model variants"

        # Should have Claude models
        claude_models = [m for m in model_values if "claude" in m]
        assert len(claude_models) >= 2, "Should have multiple Claude model variants"

        # Should have Gemini models
        gemini_models = [m for m in model_values if "gemini" in m]
        assert len(gemini_models) >= 2, "Should have multiple Gemini model variants"

    def test_data_consistency(self):
        """Test consistency between different data structures."""
        # Score range should be compatible with rating validation
        assert SCORE_RANGE[0] >= 1, "Minimum score should be at least 1"
        assert SCORE_RANGE[1] <= 5, "Maximum score should be at most 5"

        # Confidence range should be standard probability range
        assert CONFIDENCE_RANGE[0] == 0.0, "Minimum confidence should be 0"
        assert CONFIDENCE_RANGE[1] == 1.0, "Maximum confidence should be 1"

        # Weight tolerance should be reasonable for floating point precision
        assert 0.001 <= WEIGHT_SUM_TOLERANCE <= 0.1, (
            "Weight tolerance should be reasonable"
        )


if __name__ == "__main__":
    # Run unit tests directly
    pytest.main([__file__, "-v"])
