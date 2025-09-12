"""Integration tests for LLM adapters module.

This module contains integration tests that make real API calls to test
the actual functionality of sector analysis adapters. Deep research adapters
are marked as @mark.slow due to their longer execution times and higher costs.

These tests require valid API keys to be configured in the environment.
Tests will fail if the required models are not available.
"""

import pytest
import os

from sector_committee.data_models import SectorRequest, ModelResult
from sector_committee.llm_models import ModelName
from sector_committee.scoring.llm_adapters import (
    SectorAnalysisError,
    StandardSectorAdapter,
    DeepResearchSectorAdapter,
    SectorAdapterFactory,
)


# Skip all tests if no API keys are available
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="API keys not configured - skipping integration tests",
)


class TestStandardSectorAdapterIntegration:
    """Integration tests for StandardSectorAdapter.

    These tests use real API calls but should be relatively fast.
    """

    @pytest.mark.asyncio
    async def test_gpt4_standard_adapter_analysis(self):
        """Test GPT-4 standard adapter analysis."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_GPT4, timeout_seconds=60
        )
        assert isinstance(adapter, StandardSectorAdapter)

        request = SectorRequest(sector="Utilities", horizon_weeks=4)
        result = await adapter.analyze_sector(request)

        assert isinstance(result, ModelResult)
        assert result.model == ModelName.OPENAI_GPT4
        assert result.success is True
        assert result.latency_ms > 0
        assert isinstance(result.data, dict)
        assert "rating" in result.data
        assert 1 <= result.data["rating"] <= 5

    @pytest.mark.asyncio
    async def test_gpt4o_structured_adapter_analysis(self):
        """Test GPT-4o structured adapter analysis."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_GPT4O_STRUCTURED, timeout_seconds=60
        )
        assert isinstance(adapter, StandardSectorAdapter)

        request = SectorRequest(sector="Information Technology", horizon_weeks=4)
        result = await adapter.analyze_sector(request)

        assert isinstance(result, ModelResult)
        assert result.model == ModelName.OPENAI_GPT4O_STRUCTURED
        assert result.success is True
        assert result.latency_ms > 0
        assert isinstance(result.data, dict)
        assert "rating" in result.data
        assert 1 <= result.data["rating"] <= 5

        # Structured outputs should have all required fields
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
            assert field in result.data

    @pytest.mark.asyncio
    async def test_standard_adapter_multiple_sectors(self):
        """Test standard adapter with multiple different sectors."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_GPT4O, timeout_seconds=60
        )

        # Test multiple sectors to ensure consistency
        test_sectors = ["Financials", "Health Care", "Energy", "Consumer Discretionary"]

        for sector in test_sectors:
            request = SectorRequest(sector=sector, horizon_weeks=4)
            result = await adapter.analyze_sector(request)

            assert result.success is True
            assert result.model == ModelName.OPENAI_GPT4O
            assert isinstance(result.data, dict)
            assert "rating" in result.data
            assert 1 <= result.data["rating"] <= 5
            assert result.data["summary"]  # Should have non-empty summary


class TestDeepResearchSectorAdapterIntegration:
    """Integration tests for DeepResearchSectorAdapter.

    These tests are marked as @mark.slow due to longer execution times
    and higher resource usage.
    """

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_o3_deep_research_adapter_analysis(self):
        """Test o3-deep-research adapter analysis."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_O3_DEEP_RESEARCH, timeout_seconds=300
        )
        assert isinstance(adapter, DeepResearchSectorAdapter)

        request = SectorRequest(sector="Financials", horizon_weeks=4)
        result = await adapter.analyze_sector(request)

        assert isinstance(result, ModelResult)
        assert result.model == ModelName.OPENAI_O3_DEEP_RESEARCH
        assert result.success is True
        assert result.latency_ms > 0
        assert isinstance(result.data, dict)
        assert "rating" in result.data
        assert 1 <= result.data["rating"] <= 5

        # Deep research should have comprehensive analysis
        assert len(result.data.get("rationale", [])) > 0
        assert result.data.get("confidence", 0) > 0

        # Should have all required structured fields
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
            assert field in result.data

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_o4_mini_deep_research_adapter_analysis(self):
        """Test o4-mini-deep-research adapter analysis."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH, timeout_seconds=300
        )
        assert isinstance(adapter, DeepResearchSectorAdapter)

        request = SectorRequest(sector="Health Care", horizon_weeks=8)
        result = await adapter.analyze_sector(request)

        assert isinstance(result, ModelResult)
        assert result.model == ModelName.OPENAI_O4_MINI_DEEP_RESEARCH
        assert result.success is True
        assert result.latency_ms > 0
        assert isinstance(result.data, dict)
        assert "rating" in result.data
        assert 1 <= result.data["rating"] <= 5

        # Deep research should have comprehensive analysis
        assert len(result.data.get("rationale", [])) > 0
        assert result.data.get("confidence", 0) > 0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_deep_research_two_stage_pipeline(self):
        """Test that deep research adapter uses proper two-stage pipeline."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_O3_DEEP_RESEARCH, timeout_seconds=300
        )
        request = SectorRequest(sector="Energy", horizon_weeks=4)

        # Test that the adapter correctly uses the underlying two-stage model
        assert isinstance(adapter, DeepResearchSectorAdapter)
        assert adapter.llm_model.reasoning_model == "gpt-4o"  # o3 maps to gpt-4o
        assert adapter.llm_model.structured_model == "gpt-4o-2024-08-06"

        result = await adapter.analyze_sector(request)

        assert result.success is True
        assert isinstance(result.data, dict)

        # Verify structured output compliance
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
            assert field in result.data, f"Missing required field: {field}"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_deep_research_reference_processing(self):
        """Test that deep research adapter properly processes references."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH, timeout_seconds=300
        )
        request = SectorRequest(sector="Technology", horizon_weeks=4)

        result = await adapter.analyze_sector(request)

        assert result.success is True
        references = result.data.get("references", [])

        if references:  # If references are provided
            for ref in references:
                # Each reference should have required fields
                assert "url" in ref
                assert "title" in ref
                assert "description" in ref
                assert "accessed_at" in ref
                assert "accessible" in ref


class TestSectorAdapterFactoryIntegration:
    """Integration tests for SectorAdapterFactory."""

    def test_factory_adapter_type_selection(self):
        """Test factory correctly selects adapter types."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        # Standard models should get StandardSectorAdapter
        standard_adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_GPT4, timeout_seconds=60
        )
        assert isinstance(standard_adapter, StandardSectorAdapter)

        # Deep research models should get DeepResearchSectorAdapter
        deep_adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_O3_DEEP_RESEARCH, timeout_seconds=300
        )
        assert isinstance(deep_adapter, DeepResearchSectorAdapter)

    def test_factory_default_adapter(self):
        """Test factory default adapter creation."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        adapter = SectorAdapterFactory.create_default_adapter(timeout_seconds=120)

        # Should prefer deep research adapters
        assert isinstance(adapter, (StandardSectorAdapter, DeepResearchSectorAdapter))
        assert adapter.get_model_name() in [
            ModelName.OPENAI_O3_DEEP_RESEARCH,
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH,
            ModelName.OPENAI_GPT4O,  # Fallback
            ModelName.OPENAI_GPT4,  # Fallback
        ]

    @pytest.mark.asyncio
    async def test_factory_adapter_functionality(self):
        """Test that factory-created adapters work correctly."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        # Test with standard model
        standard_adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_GPT4O, timeout_seconds=60
        )
        request = SectorRequest(sector="Materials", horizon_weeks=4)

        result = await standard_adapter.analyze_sector(request)
        assert result.success is True
        assert result.model == ModelName.OPENAI_GPT4O

    def test_factory_no_api_keys(self):
        """Test factory behavior with no API keys."""
        with pytest.MonkeyPatch().context() as m:
            # Remove all API keys
            m.delenv("OPENAI_API_KEY", raising=False)
            m.delenv("ANTHROPIC_API_KEY", raising=False)
            m.delenv("GOOGLE_API_KEY", raising=False)

            # Should fail to create any adapter
            with pytest.raises(SectorAnalysisError) as exc_info:
                SectorAdapterFactory.create_adapter(ModelName.OPENAI_GPT4)

            assert "Adapter creation failed" in str(exc_info.value)

            # Should also fail to create default adapter
            with pytest.raises(SectorAnalysisError) as exc_info:
                SectorAdapterFactory.create_default_adapter()

            assert "Default adapter creation failed" in str(exc_info.value)


class TestAdapterErrorHandlingIntegration:
    """Integration tests for adapter error handling."""

    @pytest.mark.asyncio
    async def test_invalid_sector_name_handling(self):
        """Test handling of invalid sector names."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_GPT4O, timeout_seconds=60
        )

        # Create request with invalid sector name
        request = SectorRequest(sector="Invalid Sector Name", horizon_weeks=4)

        # Invalid sectors may cause validation failures due to empty rationale/references
        # This is expected behavior - garbage in, validation failure out
        try:
            result = await adapter.analyze_sector(request)
            # If it succeeds, it should have valid data
            assert result.success is True
            assert (
                result.data["confidence"] < 0.7
            )  # Should have low confidence for invalid sectors
        except SectorAnalysisError as e:
            # Schema validation failure is acceptable for invalid sectors
            assert "Schema validation failed" in str(
                e
            ) or "Invalid response format" in str(e)

    @pytest.mark.asyncio
    async def test_timeout_handling_standard_adapter(self):
        """Test timeout handling for standard adapter."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        # Create adapter with very short timeout
        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_GPT4O, timeout_seconds=1
        )
        request = SectorRequest(sector="Information Technology", horizon_weeks=4)

        # Should either succeed quickly or timeout gracefully
        try:
            result = await adapter.analyze_sector(request)
            assert result.success is True
        except SectorAnalysisError as e:
            # Timeout is acceptable for this test
            assert "timed out" in str(e) or "timeout" in str(e).lower()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_timeout_handling_deep_research_adapter(self):
        """Test timeout handling for deep research adapter."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        # Create adapter with very short timeout for deep research
        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH, timeout_seconds=1
        )
        request = SectorRequest(sector="Technology", horizon_weeks=4)

        # Should either timeout gracefully or succeed very quickly
        try:
            result = await adapter.analyze_sector(request)
            # If it succeeds with such a short timeout, that's also acceptable
            assert result.success is True
        except SectorAnalysisError as e:
            # Timeout is acceptable for this test
            assert (
                "timed out" in str(e)
                or "timeout" in str(e).lower()
                or "failed" in str(e).lower()
            )


class TestAdapterPerformanceIntegration:
    """Integration tests for adapter performance characteristics."""

    @pytest.mark.asyncio
    async def test_standard_adapter_performance(self):
        """Test that standard adapters perform within reasonable bounds."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_GPT4O, timeout_seconds=60
        )
        request = SectorRequest(sector="Utilities", horizon_weeks=4)

        result = await adapter.analyze_sector(request)

        assert result.success is True
        assert result.latency_ms < 60000  # Should complete within 1 minute
        assert result.latency_ms > 0  # Should take some time

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_deep_research_adapter_performance(self):
        """Test that deep research adapters perform within reasonable bounds."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH, timeout_seconds=300
        )
        request = SectorRequest(sector="Technology", horizon_weeks=4)

        result = await adapter.analyze_sector(request)

        assert result.success is True
        assert result.latency_ms < 300000  # Should complete within 5 minutes
        assert result.latency_ms > 1000  # Should take reasonable time for deep research

    @pytest.mark.asyncio
    async def test_adapter_consistency(self):
        """Test that adapters produce consistent results."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_GPT4O, timeout_seconds=60
        )
        request = SectorRequest(sector="Financials", horizon_weeks=4)

        # Run analysis twice to check consistency
        result1 = await adapter.analyze_sector(request)
        result2 = await adapter.analyze_sector(request)

        assert result1.success is True
        assert result2.success is True
        assert result1.model == result2.model

        # Results should be similar (within 1 point for rating)
        rating_diff = abs(result1.data["rating"] - result2.data["rating"])
        assert rating_diff <= 1


class TestAdapterValidationIntegration:
    """Integration tests for adapter validation."""

    @pytest.mark.asyncio
    async def test_structured_output_validation(self):
        """Test that adapters produce valid structured outputs."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_GPT4O_STRUCTURED, timeout_seconds=60
        )
        request = SectorRequest(sector="Communication Services", horizon_weeks=4)

        result = await adapter.analyze_sector(request)

        assert result.success is True
        data = result.data

        # Validate structure
        assert isinstance(data["rating"], int)
        assert isinstance(data["summary"], str)
        assert isinstance(data["sub_scores"], dict)
        assert isinstance(data["weights"], dict)
        assert isinstance(data["weighted_score"], (int, float))
        assert isinstance(data["rationale"], list)
        assert isinstance(data["references"], list)
        assert isinstance(data["confidence"], (int, float))

        # Validate ranges
        assert 1 <= data["rating"] <= 5
        assert 0 <= data["confidence"] <= 1
        assert len(data["summary"]) > 10  # Should have meaningful summary

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_deep_research_comprehensive_output(self):
        """Test that deep research adapters produce comprehensive outputs."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_O3_DEEP_RESEARCH, timeout_seconds=300
        )
        request = SectorRequest(sector="Energy", horizon_weeks=4)

        result = await adapter.analyze_sector(request)

        assert result.success is True
        data = result.data

        # Deep research should have more comprehensive content
        assert len(data["rationale"]) >= 3  # Should have multiple rationale points
        assert len(data["summary"]) > 50  # Should have detailed summary

        # Should have meaningful sub-scores
        sub_scores = data["sub_scores"]
        assert len(sub_scores) >= 3  # Should analyze multiple dimensions

        # Weights should sum to approximately 1.0
        weights = data["weights"]
        weight_sum = sum(weights.values())
        assert abs(weight_sum - 1.0) < 0.01


class TestBackwardCompatibilityIntegration:
    """Integration tests for backward compatibility through factory."""

    @pytest.mark.asyncio
    async def test_factory_backward_compatibility(self):
        """Test that the updated factory maintains backward compatibility."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        # Import the legacy factory interface
        from sector_committee.scoring.factory import ModelFactory, ResearchClient

        # Should be able to create client using old interface
        client = ModelFactory.create(ModelName.OPENAI_GPT4O, timeout_seconds=60)
        assert isinstance(client, ResearchClient)

        # Should work with old interface
        request = SectorRequest(sector="Consumer Staples", horizon_weeks=4)
        result = await client.analyze_sector(request)

        assert isinstance(result, ModelResult)
        assert result.success is True
        assert client.get_model_name() == ModelName.OPENAI_GPT4O

    @pytest.mark.asyncio
    async def test_factory_default_client_compatibility(self):
        """Test default client creation through factory."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        from sector_committee.scoring.factory import ModelFactory

        # Should be able to create default client
        client = ModelFactory.create_default_client(timeout_seconds=120)

        # Should work with old interface
        request = SectorRequest(sector="Industrials", horizon_weeks=4)
        result = await client.analyze_sector(request)

        assert result.success is True

        # Should prefer deep research models for default
        model_name = client.get_model_name()
        assert model_name in [
            ModelName.OPENAI_O3_DEEP_RESEARCH,
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH,
            ModelName.OPENAI_GPT4O,  # Fallback
            ModelName.OPENAI_GPT4,  # Fallback
        ]

    def test_factory_supported_models_compatibility(self):
        """Test that factory supported models function works."""
        from sector_committee.scoring.factory import ModelFactory

        models = ModelFactory.get_supported_models()
        assert len(models) >= 4
        assert ModelName.OPENAI_GPT4 in models
        assert ModelName.OPENAI_O3_DEEP_RESEARCH in models


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_end_to_end_standard_flow(self):
        """Test complete end-to-end flow with standard adapter."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        # Create adapter through factory
        adapter = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_GPT4O_STRUCTURED, timeout_seconds=60
        )

        # Create comprehensive request
        request = SectorRequest(sector="Information Technology", horizon_weeks=4)

        # Execute analysis
        result = await adapter.analyze_sector(request)

        # Verify complete flow
        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.latency_ms > 0
        assert result.timestamp_utc

        # Verify all scoring components are present
        assert "rating" in result.data
        assert "sub_scores" in result.data
        assert "weights" in result.data
        assert "weighted_score" in result.data
        assert "confidence" in result.data

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_end_to_end_deep_research_flow(self):
        """Test complete end-to-end flow with deep research adapter."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        # Create deep research adapter through factory
        adapter = SectorAdapterFactory.create_default_adapter(timeout_seconds=300)

        # Create comprehensive request
        request = SectorRequest(sector="Health Care", horizon_weeks=8)

        # Execute analysis
        result = await adapter.analyze_sector(request)

        # Verify complete flow
        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.latency_ms > 0

        # Deep research should have comprehensive output
        assert len(result.data["rationale"]) >= 2
        assert result.data["confidence"] > 0.5  # Should have decent confidence

        # Should have processed any references
        references = result.data.get("references", [])
        if references:
            for ref in references:
                assert "accessible" in ref
                assert "accessed_at" in ref

    @pytest.mark.asyncio
    async def test_multiple_adapters_concurrent(self):
        """Test multiple adapters running concurrently."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        import asyncio

        # Create multiple adapters
        adapter1 = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_GPT4O, timeout_seconds=60
        )
        adapter2 = SectorAdapterFactory.create_adapter(
            ModelName.OPENAI_GPT4O, timeout_seconds=60
        )

        # Create different requests
        request1 = SectorRequest(sector="Utilities", horizon_weeks=4)
        request2 = SectorRequest(sector="Materials", horizon_weeks=4)

        # Run concurrently
        results = await asyncio.gather(
            adapter1.analyze_sector(request1), adapter2.analyze_sector(request2)
        )

        # Both should succeed
        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].model == ModelName.OPENAI_GPT4O
        assert results[1].model == ModelName.OPENAI_GPT4O
