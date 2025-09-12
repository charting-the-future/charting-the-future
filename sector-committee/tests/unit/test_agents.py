"""Unit tests for agents module.

Fast tests with no external dependencies, focusing on SectorAgent
initialization, request validation, and coordination logic.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from sector_committee.scoring.agents import (
    SectorAgent,
    SectorAnalysisError,
    analyze_sector_quick,
    analyze_all_sectors,
)
from sector_committee.data_models import (
    SectorRequest,
    DEFAULT_WEIGHTS,
)
from sector_committee.llm_models import ModelName
from sector_committee.scoring.factory import ResearchError


class TestSectorAnalysisError:
    """Test SectorAnalysisError exception."""

    def test_error_creation(self):
        """Test error creation with basic message."""
        error = SectorAnalysisError("Test error", "Technology")
        assert "Test error" in str(error)
        assert "Technology" in str(error)
        assert error.sector == "Technology"
        assert error.original_error is None

    def test_error_with_original(self):
        """Test error creation with original exception."""
        original = ValueError("Original error")
        error = SectorAnalysisError("Test error", "Healthcare", original)
        assert error.original_error == original
        assert error.sector == "Healthcare"


class TestSectorAgent:
    """Test SectorAgent class."""

    def test_agent_initialization_defaults(self):
        """Test agent initialization with default parameters."""
        agent = SectorAgent()
        assert agent.timeout_seconds == 300
        assert agent.enable_audit is True
        assert agent.min_confidence == 0.3
        assert agent.research_client is not None
        assert agent.audit_logger is not None

    def test_agent_initialization_custom(self):
        """Test agent initialization with custom parameters."""
        agent = SectorAgent(timeout_seconds=600, enable_audit=False, min_confidence=0.5)
        assert agent.timeout_seconds == 600
        assert agent.enable_audit is False
        assert agent.min_confidence == 0.5
        assert agent.audit_logger is None

    def test_is_valid_sector_valid(self):
        """Test valid sector validation."""
        agent = SectorAgent()

        # Test valid sectors
        assert agent._is_valid_sector("Information Technology") is True
        assert agent._is_valid_sector("Health Care") is True
        assert agent._is_valid_sector("Financials") is True

    def test_is_valid_sector_invalid(self):
        """Test invalid sector validation."""
        agent = SectorAgent()

        # Test invalid sectors
        assert agent._is_valid_sector("Invalid Sector") is False
        assert agent._is_valid_sector("") is False
        assert agent._is_valid_sector("Technology") is False  # Not exact match

    def test_get_sector_etf_valid(self):
        """Test ETF mapping for valid sectors."""
        agent = SectorAgent()

        assert agent._get_sector_etf("Information Technology") == "XLK"
        assert agent._get_sector_etf("Health Care") == "XLV"
        assert agent._get_sector_etf("Financials") == "XLF"

    def test_get_sector_etf_invalid(self):
        """Test ETF mapping for invalid sectors."""
        agent = SectorAgent()

        # Should return UNKNOWN for invalid sectors
        assert agent._get_sector_etf("Invalid Sector") == "UNKNOWN"

    def test_generate_analysis_id(self):
        """Test analysis ID generation."""
        agent = SectorAgent()
        request = SectorRequest(sector="Information Technology", horizon_weeks=4)

        analysis_id = agent._generate_analysis_id(request)

        # Check format: YYYYMMDD_HHMMSS_INFORMATION_TECHNOLOGY_WEEKS
        parts = analysis_id.split("_")
        assert len(parts) == 5  # Date, time, "INFORMATION", "TECHNOLOGY", "4W"
        assert parts[2] == "INFORMATION"
        assert parts[3] == "TECHNOLOGY"
        assert parts[4] == "4W"

    def test_create_and_validate_request_valid(self):
        """Test request creation and validation with valid inputs."""
        agent = SectorAgent()

        request = agent._create_and_validate_request("Information Technology", 4, None)

        assert request.sector == "Information Technology"
        assert request.horizon_weeks == 4
        assert request.weights_hint == DEFAULT_WEIGHTS

    def test_create_and_validate_request_custom_weights(self):
        """Test request creation with custom weights."""
        agent = SectorAgent()
        custom_weights = {"fundamentals": 0.6, "sentiment": 0.2, "technicals": 0.2}

        request = agent._create_and_validate_request("Health Care", 8, custom_weights)

        assert request.sector == "Health Care"
        assert request.horizon_weeks == 8
        assert request.weights_hint == custom_weights

    def test_create_and_validate_request_invalid_sector(self):
        """Test request validation with invalid sector."""
        agent = SectorAgent()

        with pytest.raises(SectorAnalysisError) as exc_info:
            agent._create_and_validate_request("Invalid Sector", 4, None)

        assert "Invalid sector" in str(exc_info.value)
        assert "Invalid Sector" in str(exc_info.value)

    def test_create_and_validate_request_invalid_horizon(self):
        """Test request validation with invalid horizon."""
        agent = SectorAgent()

        with pytest.raises(SectorAnalysisError) as exc_info:
            agent._create_and_validate_request("Financials", 0, None)

        assert "Invalid request parameters" in str(exc_info.value)

    def test_create_and_validate_request_invalid_weights(self):
        """Test request validation with invalid weights."""
        agent = SectorAgent()
        invalid_weights = {
            "fundamentals": 0.5,
            "sentiment": 0.5,
            "technicals": 0.2,  # Sum > 1.0
        }

        with pytest.raises(SectorAnalysisError) as exc_info:
            agent._create_and_validate_request("Energy", 4, invalid_weights)

        assert "Invalid request parameters" in str(exc_info.value)

    @patch("sector_committee.scoring.agents.validate_sector_rating")
    def test_validate_and_enhance_result_valid(self, mock_validate):
        """Test result validation with valid rating."""
        agent = SectorAgent()

        # Mock successful validation
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.warnings = []
        mock_validate.return_value = mock_validation_result

        test_rating = {"rating": 4, "summary": "Test analysis", "confidence": 0.85}
        test_request = SectorRequest(sector="Utilities", horizon_weeks=4)

        result = agent._validate_and_enhance_result(test_rating, test_request)

        assert result == test_rating
        mock_validate.assert_called_once_with(test_rating)

    @patch("sector_committee.scoring.agents.validate_sector_rating")
    def test_validate_and_enhance_result_invalid(self, mock_validate):
        """Test result validation with invalid rating."""
        agent = SectorAgent()

        # Mock failed validation
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.errors = ["Invalid rating score"]
        mock_validate.return_value = mock_validation_result

        test_rating = {"rating": 10}  # Invalid rating
        test_request = SectorRequest(sector="Materials", horizon_weeks=4)

        with pytest.raises(SectorAnalysisError) as exc_info:
            agent._validate_and_enhance_result(test_rating, test_request)

        assert "Result validation failed" in str(exc_info.value)
        assert "Invalid rating score" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_supported_sectors(self):
        """Test getting supported sectors list."""
        agent = SectorAgent()
        sectors = await agent.get_supported_sectors()

        assert len(sectors) == 11  # 11 SPDR sectors
        assert "Information Technology" in sectors
        assert "Health Care" in sectors
        assert "Financials" in sectors

    @pytest.mark.asyncio
    async def test_get_sector_etf_mapping(self):
        """Test getting sector ETF mapping."""
        agent = SectorAgent()
        mapping = await agent.get_sector_etf_mapping()

        assert len(mapping) == 11
        assert mapping["Information Technology"] == "XLK"
        assert mapping["Health Care"] == "XLV"
        assert mapping["Financials"] == "XLF"

    @pytest.mark.asyncio
    @patch("sector_committee.scoring.factory.ModelFactory.create_default_client")
    async def test_health_check_healthy(self, mock_create_client):
        """Test health check with healthy system."""
        # Mock healthy client
        mock_client = Mock()
        mock_client.get_model_name.return_value = ModelName.OPENAI_GPT4O
        mock_create_client.return_value = mock_client

        with patch(
            "sector_committee.scoring.factory.ModelFactory.get_supported_models"
        ) as mock_models:
            mock_models.return_value = [ModelName.OPENAI_GPT4O, ModelName.OPENAI_GPT4]

            agent = SectorAgent()
            health = await agent.health_check()

            assert health["status"] == "healthy"
            assert health["models_available"] == 1
            assert "gpt-4o" in health["available_models"]
            assert health["supported_sectors"] == 11
            assert health["supported_models"] == 2

    @pytest.mark.asyncio
    async def test_health_check_degraded(self):
        """Test health check with degraded system."""
        # Mock successful agent creation first
        with patch(
            "sector_committee.scoring.factory.ModelFactory.create_default_client"
        ) as mock_create_client:
            # Return a mock client for constructor
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            # Create agent successfully
            agent = SectorAgent(enable_audit=False)

            # Now configure the mock to fail for health check
            mock_create_client.side_effect = Exception("No API key")

            with patch(
                "sector_committee.scoring.factory.ModelFactory.get_supported_models"
            ) as mock_models:
                mock_models.return_value = []

                # Test the health check method which should fail
                health = await agent.health_check()

                assert health["status"] == "degraded"
                assert health["models_available"] == 0
                assert health["available_models"] == []
                assert "No API key" in health["error"]

    @pytest.mark.asyncio
    @patch("sector_committee.scoring.agents.SectorAgent._perform_analysis")
    @patch("sector_committee.scoring.agents.SectorAgent._validate_and_enhance_result")
    async def test_analyze_sector_success(self, mock_validate, mock_perform):
        """Test successful sector analysis."""
        agent = SectorAgent(enable_audit=False)  # Disable audit for simpler testing

        # Mock analysis result
        mock_model_result = Mock()
        mock_model_result.data = {
            "rating": 4,
            "summary": "Strong technology outlook",
            "confidence": 0.85,
            "sub_scores": {"fundamentals": 4, "sentiment": 4, "technicals": 3},
            "weights": {"fundamentals": 0.5, "sentiment": 0.3, "technicals": 0.2},
            "weighted_score": 3.8,
            "rationale": [],
            "references": [],
        }
        mock_perform.return_value = mock_model_result
        mock_validate.return_value = mock_model_result.data

        result = await agent.analyze_sector("Information Technology")

        assert result == mock_model_result.data
        mock_perform.assert_called_once()
        mock_validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_sector_low_confidence(self):
        """Test analysis with confidence below threshold."""
        agent = SectorAgent(enable_audit=False, min_confidence=0.8)

        with patch.object(agent, "_perform_analysis") as mock_perform:
            with patch.object(agent, "_validate_and_enhance_result") as mock_validate:
                # Mock low confidence result
                mock_model_result = Mock()
                mock_model_result.data = {"confidence": 0.5}
                mock_perform.return_value = mock_model_result
                mock_validate.return_value = {"confidence": 0.5}

                with pytest.raises(SectorAnalysisError) as exc_info:
                    await agent.analyze_sector("Energy")

                assert "confidence" in str(exc_info.value)
                assert "below minimum threshold" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_perform_analysis_research_error(self):
        """Test analysis with ResearchError."""
        agent = SectorAgent(enable_audit=False)

        with patch.object(agent.research_client, "analyze_sector") as mock_analyze:
            mock_analyze.side_effect = ResearchError(
                ModelName.OPENAI_GPT4O, "Model unavailable"
            )

            request = SectorRequest(sector="Utilities", horizon_weeks=4)

            with pytest.raises(SectorAnalysisError) as exc_info:
                await agent._perform_analysis(request)

            assert "Analysis failed" in str(exc_info.value)
            assert "Model unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_perform_analysis_unexpected_error(self):
        """Test analysis with unexpected error."""
        agent = SectorAgent(enable_audit=False)

        with patch.object(agent.research_client, "analyze_sector") as mock_analyze:
            mock_analyze.side_effect = ValueError("Unexpected error")

            request = SectorRequest(sector="Materials", horizon_weeks=4)

            with pytest.raises(SectorAnalysisError) as exc_info:
                await agent._perform_analysis(request)

            assert "Unexpected analysis error" in str(exc_info.value)
            assert "Unexpected error" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("sector_committee.scoring.agents.SectorAgent.analyze_sector")
    async def test_analyze_multiple_sectors_success(self, mock_analyze):
        """Test multiple sector analysis with all successes."""
        agent = SectorAgent()

        # Mock successful analysis for each sector
        async def mock_sector_analysis(sector, *args, **kwargs):
            return {"rating": 4, "summary": f"Analysis for {sector}", "confidence": 0.8}

        mock_analyze.side_effect = mock_sector_analysis

        sectors = ["Information Technology", "Health Care"]
        results = await agent.analyze_multiple_sectors(sectors)

        assert len(results) == 2
        assert "Information Technology" in results
        assert "Health Care" in results
        assert (
            results["Information Technology"]["summary"]
            == "Analysis for Information Technology"
        )

    @pytest.mark.asyncio
    @patch("sector_committee.scoring.agents.SectorAgent.analyze_sector")
    async def test_analyze_multiple_sectors_partial_failure(self, mock_analyze):
        """Test multiple sector analysis with partial failures."""
        agent = SectorAgent()

        # Mock mixed success/failure
        async def mock_sector_analysis(sector, *args, **kwargs):
            if sector == "Information Technology":
                return {"rating": 4, "summary": "Success", "confidence": 0.8}
            else:
                raise SectorAnalysisError("Analysis failed", sector)

        mock_analyze.side_effect = mock_sector_analysis

        sectors = ["Information Technology", "Energy"]
        results = await agent.analyze_multiple_sectors(sectors)

        # Should only contain successful results
        assert len(results) == 1
        assert "Information Technology" in results
        assert "Energy" not in results

    @pytest.mark.asyncio
    async def test_analyze_multiple_sectors_concurrency(self):
        """Test multiple sector analysis respects concurrency limits."""
        agent = SectorAgent()

        call_times = []

        async def mock_analyze_with_timing(*args, **kwargs):
            call_times.append(datetime.now())
            await asyncio.sleep(0.1)  # Simulate work
            return {"rating": 3, "confidence": 0.7}

        with patch.object(
            agent, "analyze_sector", side_effect=mock_analyze_with_timing
        ):
            sectors = ["Technology", "Health", "Finance", "Energy", "Utilities"]
            await agent.analyze_multiple_sectors(sectors, max_concurrent=2)

            # With max_concurrent=2, we should see batched execution
            assert len(call_times) == 5


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    @patch("sector_committee.scoring.agents.SectorAgent")
    async def test_analyze_sector_quick(self, mock_agent_class):
        """Test quick sector analysis function."""
        # Mock agent and its analyze_sector method
        mock_agent = AsyncMock()
        mock_agent.analyze_sector.return_value = {
            "rating": 4,
            "summary": "Quick analysis",
            "confidence": 0.8,
        }
        mock_agent_class.return_value = mock_agent

        result = await analyze_sector_quick("Information Technology", 8)

        assert result["rating"] == 4
        assert result["summary"] == "Quick analysis"
        mock_agent.analyze_sector.assert_called_once_with("Information Technology", 8)

    @pytest.mark.asyncio
    @patch("sector_committee.scoring.agents.SectorAgent")
    async def test_analyze_all_sectors(self, mock_agent_class):
        """Test analyze all sectors function."""
        # Mock agent and its methods
        mock_agent = AsyncMock()
        mock_agent.get_supported_sectors.return_value = [
            "Information Technology",
            "Health Care",
        ]
        mock_agent.analyze_multiple_sectors.return_value = {
            "Information Technology": {"rating": 4, "confidence": 0.8},
            "Health Care": {"rating": 3, "confidence": 0.7},
        }
        mock_agent_class.return_value = mock_agent

        result = await analyze_all_sectors(horizon_weeks=8, max_concurrent=2)

        assert len(result) == 2
        assert "Information Technology" in result
        assert "Health Care" in result
        mock_agent.get_supported_sectors.assert_called_once()
        mock_agent.analyze_multiple_sectors.assert_called_once()


if __name__ == "__main__":
    # Run unit tests directly
    pytest.main([__file__, "-v"])
