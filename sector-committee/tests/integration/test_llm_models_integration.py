"""Integration tests for LLM models module.

This module contains integration tests that make real API calls to test
the actual functionality of LLM models. Deep research models are marked
as @mark.slow due to their longer execution times and higher costs.

These tests require valid API keys to be configured in the environment.
Tests will be skipped if the required API keys are not available.

Note: These tests focus on the pure LLM interface (generate_text, generate_structured,
generate_enhanced_reasoning). Sector-specific analysis tests are in test_llm_adapters_integration.py.
"""

import pytest
import os

from sector_committee.llm_models import (
    ModelName,
    LLMError,
    OpenAIProvider,
    OpenAIStandardModel,
    OpenAITwoStageModel,
    ClaudeProvider,
    GeminiProvider,
    ModelRegistry,
    LLMModelFactory,
    create_model,
    get_supported_models,
    get_available_models,
    create_default_model,
)


# Skip all tests if no API keys are available
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="API keys not configured - skipping integration tests",
)


class TestOpenAIStandardModelIntegration:
    """Integration tests for OpenAI standard models.

    These tests use real API calls but should be relatively fast.
    Tests focus on the pure LLM interface.
    """

    @pytest.mark.asyncio
    async def test_gpt4_text_generation(self):
        """Test GPT-4 standard model text generation."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        model = OpenAIStandardModel(ModelName.OPENAI_GPT4, timeout_seconds=60)

        system_prompt = "You are a financial analyst. Provide clear, concise analysis."
        user_prompt = "Analyze the current outlook for the Utilities sector over the next 4 weeks."

        response = await model.generate_text(system_prompt, user_prompt)

        assert response.model == ModelName.OPENAI_GPT4
        assert response.latency_ms > 0
        assert isinstance(response.content, str)
        assert len(response.content) > 50  # Should have substantial content
        assert "util" in response.content.lower()  # Should mention utilities

    @pytest.mark.asyncio
    async def test_gpt4o_structured_generation(self):
        """Test GPT-4o with structured outputs."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        model = OpenAIStandardModel(
            ModelName.OPENAI_GPT4O_STRUCTURED, timeout_seconds=60
        )

        system_prompt = "You are a financial analyst. Provide structured analysis."
        user_prompt = "Analyze the Information Technology sector for the next 4 weeks."

        # Simple schema for testing structured output
        schema = {
            "type": "object",
            "properties": {
                "sector": {"type": "string"},
                "outlook": {"type": "string"},
                "rating": {"type": "integer", "minimum": 1, "maximum": 5},
                "key_factors": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["sector", "outlook", "rating", "key_factors"],
            "additionalProperties": False,
        }

        response = await model.generate_structured(system_prompt, user_prompt, schema)

        assert response.model == ModelName.OPENAI_GPT4O_STRUCTURED
        assert response.latency_ms > 0
        assert isinstance(response.content, dict)

        # Verify structured output compliance
        assert "sector" in response.content
        assert "outlook" in response.content
        assert "rating" in response.content
        assert "key_factors" in response.content
        assert isinstance(response.content["rating"], int)
        assert 1 <= response.content["rating"] <= 5
        assert isinstance(response.content["key_factors"], list)


class TestOpenAIDeepResearchModelIntegration:
    """Integration tests for OpenAI deep research models.

    These tests are marked as @mark.slow due to longer execution times
    and higher resource usage. Tests focus on two-stage capabilities.
    """

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_o3_enhanced_reasoning(self):
        """Test o3-deep-research enhanced reasoning generation."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        model = OpenAITwoStageModel(
            ModelName.OPENAI_O3_DEEP_RESEARCH, timeout_seconds=300
        )

        system_prompt = "You are a financial analyst conducting deep research."
        user_prompt = "Provide comprehensive analysis of the Financials sector outlook over 4 weeks."

        response = await model.generate_enhanced_reasoning(system_prompt, user_prompt)

        assert response.model == ModelName.OPENAI_O3_DEEP_RESEARCH
        assert response.latency_ms > 0
        assert isinstance(response.content, str)
        assert len(response.content) > 200  # Should have substantial content
        assert "financial" in response.content.lower()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_o4_mini_structured_generation(self):
        """Test o4-mini-deep-research structured generation."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        model = OpenAITwoStageModel(
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH, timeout_seconds=300
        )

        system_prompt = (
            "You are a financial analyst. Provide structured sector analysis."
        )
        user_prompt = "Analyze Health Care sector for 8 week outlook."

        # Schema matching sector analysis format
        schema = {
            "type": "object",
            "properties": {
                "sector": {"type": "string"},
                "horizon_weeks": {"type": "integer"},
                "rating": {"type": "integer", "minimum": 1, "maximum": 5},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "summary": {"type": "string"},
                "key_factors": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "sector",
                "horizon_weeks",
                "rating",
                "confidence",
                "summary",
                "key_factors",
            ],
            "additionalProperties": False,
        }

        response = await model.generate_structured(system_prompt, user_prompt, schema)

        assert response.model == ModelName.OPENAI_O4_MINI_DEEP_RESEARCH
        assert response.latency_ms > 0
        assert isinstance(response.content, dict)

        # Verify structured output compliance
        assert "sector" in response.content
        assert "rating" in response.content
        assert "confidence" in response.content
        assert isinstance(response.content["rating"], int)
        assert 1 <= response.content["rating"] <= 5
        assert isinstance(response.content["confidence"], (int, float))
        assert 0 <= response.content["confidence"] <= 1

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_two_stage_pipeline_verification(self):
        """Test that deep research uses proper two-stage pipeline."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        model = OpenAITwoStageModel(
            ModelName.OPENAI_O3_DEEP_RESEARCH, timeout_seconds=300
        )

        # Test that the model correctly uses the mapped API model names
        assert model.reasoning_model == "gpt-4o"  # o3 maps to gpt-4o
        assert model.structured_model == "gpt-4o-2024-08-06"

        # Test two-stage generation with simple schema
        system_prompt = "You are a financial analyst."
        user_prompt = "Analyze Energy sector."

        schema = {
            "type": "object",
            "properties": {
                "analysis": {"type": "string"},
                "rating": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["analysis", "rating"],
            "additionalProperties": False,
        }

        response = await model.generate_structured(system_prompt, user_prompt, schema)

        assert response.model == ModelName.OPENAI_O3_DEEP_RESEARCH
        assert isinstance(response.content, dict)
        assert "analysis" in response.content
        assert "rating" in response.content

        # Should have metadata from two-stage process
        assert "stage1_latency_ms" in response.metadata
        assert "stage2_latency_ms" in response.metadata

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_text_generation_capability(self):
        """Test text generation on deep research models."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        model = OpenAITwoStageModel(
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH, timeout_seconds=300
        )

        system_prompt = "You are a comprehensive research analyst."
        user_prompt = "Provide detailed analysis of Technology sector trends."

        response = await model.generate_text(system_prompt, user_prompt)

        assert response.model == ModelName.OPENAI_O4_MINI_DEEP_RESEARCH
        assert isinstance(response.content, str)
        assert len(response.content) > 100
        assert "technology" in response.content.lower()


class TestOpenAIProviderIntegration:
    """Integration tests for OpenAI provider."""

    def test_provider_availability(self):
        """Test provider availability check."""
        provider = OpenAIProvider()

        if os.getenv("OPENAI_API_KEY"):
            assert provider.is_available() is True
        else:
            assert provider.is_available() is False

    @pytest.mark.asyncio
    async def test_create_and_use_standard_model(self):
        """Test creating and using a standard model."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        provider = OpenAIProvider()
        model = provider.create_model(ModelName.OPENAI_GPT4O, timeout_seconds=60)

        assert isinstance(model, OpenAIStandardModel)
        assert model.get_model_name() == ModelName.OPENAI_GPT4O

        # Test text generation
        system_prompt = "You are a financial analyst."
        user_prompt = "Analyze the Materials sector for the next 4 weeks."

        response = await model.generate_text(system_prompt, user_prompt)

        assert response.model == ModelName.OPENAI_GPT4O
        assert isinstance(response.content, str)
        assert len(response.content) > 50

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_create_and_use_deep_research_model(self):
        """Test creating and using a deep research model."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        provider = OpenAIProvider()
        model = provider.create_model(
            ModelName.OPENAI_O3_DEEP_RESEARCH, timeout_seconds=300
        )

        assert isinstance(model, OpenAITwoStageModel)
        assert model.get_model_name() == ModelName.OPENAI_O3_DEEP_RESEARCH

        # Test enhanced reasoning
        system_prompt = "You are a comprehensive research analyst."
        user_prompt = "Analyze Consumer Discretionary sector trends and outlook."

        response = await model.generate_enhanced_reasoning(system_prompt, user_prompt)

        assert response.model == ModelName.OPENAI_O3_DEEP_RESEARCH
        assert isinstance(response.content, str)
        assert len(response.content) > 100


class TestClaudeProviderIntegration:
    """Integration tests for Claude provider."""

    def test_provider_not_implemented(self):
        """Test that Claude provider is not yet implemented."""
        provider = ClaudeProvider()

        assert provider.is_available() is False
        assert len(provider.get_supported_models()) == 0

        with pytest.raises(LLMError) as exc_info:
            provider.create_model(ModelName.CLAUDE_SONNET)

        assert "not yet implemented" in str(exc_info.value)


class TestGeminiProviderIntegration:
    """Integration tests for Gemini provider."""

    def test_provider_not_implemented(self):
        """Test that Gemini provider is not yet implemented."""
        provider = GeminiProvider()

        assert provider.is_available() is False
        assert len(provider.get_supported_models()) == 0

        with pytest.raises(LLMError) as exc_info:
            provider.create_model(ModelName.GEMINI_PRO)

        assert "not yet implemented" in str(exc_info.value)


class TestModelRegistryIntegration:
    """Integration tests for ModelRegistry."""

    def test_registry_with_real_providers(self):
        """Test registry with real provider availability."""
        registry = ModelRegistry()

        # Should always have providers registered
        assert len(registry.providers) >= 3
        assert "openai" in registry.providers
        assert "claude" in registry.providers
        assert "gemini" in registry.providers

    def test_get_available_models_real(self):
        """Test getting available models with real API availability."""
        registry = ModelRegistry()
        models = registry.get_available_models()

        if os.getenv("OPENAI_API_KEY"):
            # Should have OpenAI models available
            openai_models = [
                m
                for m in models
                if m.value.startswith("gpt") or m.value.startswith("o")
            ]
            assert len(openai_models) > 0
        else:
            # Should have no models available without API keys
            assert len(models) == 0

    @pytest.mark.asyncio
    async def test_create_model_real(self):
        """Test creating a real model through registry."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        registry = ModelRegistry()
        model = registry.create_model(ModelName.OPENAI_GPT4O, timeout_seconds=60)

        assert isinstance(model, OpenAIStandardModel)

        # Test that the model actually works
        system_prompt = "You are a financial analyst."
        user_prompt = "Analyze the Utilities sector outlook."

        response = await model.generate_text(system_prompt, user_prompt)

        assert response.model == ModelName.OPENAI_GPT4O
        assert isinstance(response.content, str)
        assert len(response.content) > 50


class TestLLMModelFactoryIntegration:
    """Integration tests for LLMModelFactory."""

    def test_factory_with_real_providers(self):
        """Test factory with real provider availability."""
        factory = LLMModelFactory()

        # Should have models listed
        all_models = factory.get_supported_models()
        assert len(all_models) > 0

        available_models = factory.get_available_models()
        if os.getenv("OPENAI_API_KEY"):
            assert len(available_models) > 0
        else:
            assert len(available_models) == 0

    @pytest.mark.asyncio
    async def test_factory_create_standard_model(self):
        """Test factory creating standard model."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        factory = LLMModelFactory()
        model = factory.create_model(ModelName.OPENAI_GPT4O, timeout_seconds=60)

        assert isinstance(model, OpenAIStandardModel)

        # Test functionality
        system_prompt = "You are a financial analyst."
        user_prompt = "Analyze Real Estate sector outlook."

        response = await model.generate_text(system_prompt, user_prompt)

        assert response.model == ModelName.OPENAI_GPT4O
        assert isinstance(response.content, str)
        assert len(response.content) > 50

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_factory_create_deep_research_model(self):
        """Test factory creating deep research model."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        factory = LLMModelFactory()
        model = factory.create_model(
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH, timeout_seconds=300
        )

        assert isinstance(model, OpenAITwoStageModel)

        # Test functionality
        system_prompt = "You are a comprehensive financial analyst."
        user_prompt = "Analyze Consumer Staples sector trends."

        response = await model.generate_enhanced_reasoning(system_prompt, user_prompt)

        assert response.model == ModelName.OPENAI_O4_MINI_DEEP_RESEARCH
        assert isinstance(response.content, str)
        assert len(response.content) > 100

    @pytest.mark.asyncio
    async def test_factory_create_default_model(self):
        """Test factory creating default model."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        factory = LLMModelFactory()
        model = factory.create_default_model(timeout_seconds=120)

        # Should prefer deep research models
        assert model.get_model_name() in [
            ModelName.OPENAI_O3_DEEP_RESEARCH,
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH,
        ]

        # Test functionality
        system_prompt = "You are a financial analyst."
        user_prompt = "Analyze Industrials sector."

        response = await model.generate_text(system_prompt, user_prompt)

        assert isinstance(response.content, str)
        assert len(response.content) > 50

    def test_factory_no_api_keys(self):
        """Test factory behavior with no API keys."""
        with pytest.MonkeyPatch().context() as m:
            # Remove all API keys
            m.delenv("OPENAI_API_KEY", raising=False)
            m.delenv("ANTHROPIC_API_KEY", raising=False)
            m.delenv("GOOGLE_API_KEY", raising=False)

            factory = LLMModelFactory()

            # Should have no available models
            available_models = factory.get_available_models()
            assert len(available_models) == 0

            # Should fail to create default model
            with pytest.raises(LLMError) as exc_info:
                factory.create_default_model()

            assert "No models available" in str(exc_info.value)


class TestGlobalFunctionsIntegration:
    """Integration tests for global convenience functions."""

    @pytest.mark.asyncio
    async def test_global_create_model(self):
        """Test global create_model function."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        model = create_model(ModelName.OPENAI_GPT4O, timeout_seconds=60)
        assert isinstance(model, OpenAIStandardModel)

        # Test functionality
        system_prompt = "You are a financial analyst."
        user_prompt = "Analyze Communication Services sector outlook."

        response = await model.generate_text(system_prompt, user_prompt)

        assert response.model == ModelName.OPENAI_GPT4O
        assert isinstance(response.content, str)
        assert len(response.content) > 50

    def test_global_get_models_functions(self):
        """Test global model listing functions."""
        all_models = get_supported_models()
        assert len(all_models) > 0
        assert ModelName.OPENAI_GPT4 in all_models

        available_models = get_available_models()
        if os.getenv("OPENAI_API_KEY"):
            assert len(available_models) > 0
        else:
            assert len(available_models) == 0

    @pytest.mark.asyncio
    async def test_global_create_default_model(self):
        """Test global create_default_model function."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        model = create_default_model(timeout_seconds=120)

        # Should prefer deep research models
        assert model.get_model_name() in [
            ModelName.OPENAI_O3_DEEP_RESEARCH,
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH,
        ]

        # Test functionality
        system_prompt = "You are a financial analyst."
        user_prompt = "Analyze Financials sector outlook."

        response = await model.generate_text(system_prompt, user_prompt)

        assert isinstance(response.content, str)
        assert len(response.content) > 50


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    async def test_invalid_prompt_handling(self):
        """Test handling of unusual prompts."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        model = create_model(ModelName.OPENAI_GPT4O, timeout_seconds=60)

        # Test with unusual/empty prompts
        system_prompt = "You are a financial analyst."
        user_prompt = "Analyze this nonsensical invalid sector name: XYZ123"

        # Model should still process the request gracefully
        response = await model.generate_text(system_prompt, user_prompt)

        assert isinstance(response.content, str)
        assert len(response.content) > 10  # Should have some response

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        # Create model with very short timeout
        model = create_model(ModelName.OPENAI_GPT4O, timeout_seconds=1)

        system_prompt = "You are a financial analyst."
        user_prompt = "Provide comprehensive analysis of Information Technology sector."

        # Should either succeed quickly or timeout gracefully
        try:
            response = await model.generate_text(system_prompt, user_prompt)
            assert isinstance(response.content, str)
        except LLMError as e:
            # Timeout is acceptable for this test
            assert "timed out" in str(e) or "timeout" in str(e).lower()

    def test_missing_api_key_error(self):
        """Test proper error when API key is missing."""
        with pytest.MonkeyPatch().context() as m:
            m.delenv("OPENAI_API_KEY", raising=False)

            with pytest.raises(LLMError) as exc_info:
                OpenAIStandardModel(ModelName.OPENAI_GPT4)

            assert "Failed to initialize OpenAI client" in str(exc_info.value)


class TestPerformanceIntegration:
    """Integration tests for performance characteristics."""

    @pytest.mark.asyncio
    async def test_standard_model_performance(self):
        """Test that standard models perform within reasonable bounds."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        model = create_model(ModelName.OPENAI_GPT4O, timeout_seconds=60)

        system_prompt = "You are a financial analyst."
        user_prompt = "Analyze the Utilities sector outlook."

        response = await model.generate_text(system_prompt, user_prompt)

        assert isinstance(response.content, str)
        assert response.latency_ms < 60000  # Should complete within 1 minute
        assert response.latency_ms > 0  # Should take some time

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_deep_research_model_performance(self):
        """Test that deep research models perform within reasonable bounds."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not configured")

        model = create_model(
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH, timeout_seconds=300
        )

        system_prompt = "You are a comprehensive financial analyst."
        user_prompt = "Provide detailed analysis of Technology sector trends."

        response = await model.generate_enhanced_reasoning(system_prompt, user_prompt)

        assert isinstance(response.content, str)
        assert response.latency_ms < 300000  # Should complete within 5 minutes
        assert (
            response.latency_ms > 1000
        )  # Should take reasonable time for deep research
