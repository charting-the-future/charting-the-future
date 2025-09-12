"""Unit tests for LLM models module.

This module tests the pure LLM abstraction layer, including base classes,
provider implementations, factory pattern, and error handling.
"""

import pytest
import os
from unittest.mock import Mock, AsyncMock, patch

from sector_committee.llm_models import ModelName
from sector_committee.llm_models import (
    LLMModel,
    StandardLLMModel,
    TwoStageLLMModel,
    LLMResponse,
    LLMError,
    LLMProvider,
    OpenAIProvider,
    ClaudeProvider,
    GeminiProvider,
    OpenAIStandardModel,
    OpenAITwoStageModel,
    ModelRegistry,
    LLMModelFactory,
    create_model,
    get_supported_models,
    get_available_models,
    create_default_model,
)


class TestLLMResponse:
    """Test LLMResponse data class."""

    def test_response_creation(self):
        """Test LLMResponse creation with all fields."""
        response = LLMResponse(
            content={"test": "data"},
            model=ModelName.OPENAI_GPT4,
            latency_ms=1500,
            timestamp_utc="2024-01-01T12:00:00Z",
            metadata={"tokens": 100},
        )

        assert response.content == {"test": "data"}
        assert response.model == ModelName.OPENAI_GPT4
        assert response.latency_ms == 1500
        assert response.timestamp_utc == "2024-01-01T12:00:00Z"
        assert response.metadata == {"tokens": 100}

    def test_response_creation_minimal(self):
        """Test LLMResponse creation with minimal fields."""
        response = LLMResponse(
            content="Simple text",
            model=ModelName.OPENAI_GPT4O,
            latency_ms=500,
            timestamp_utc="2024-01-01T12:00:00Z",
            metadata={},
        )

        assert response.content == "Simple text"
        assert response.model == ModelName.OPENAI_GPT4O
        assert response.metadata == {}


class TestLLMError:
    """Test LLMError exception class."""

    def test_error_creation_basic(self):
        """Test basic error creation."""
        error = LLMError(ModelName.OPENAI_GPT4, "Test error message")

        assert error.model == ModelName.OPENAI_GPT4
        assert error.original_error is None
        assert str(error) == "gpt-4: Test error message"

    def test_error_creation_with_original(self):
        """Test error creation with original exception."""
        original = ValueError("Original error")
        error = LLMError(ModelName.OPENAI_GPT4O, "Wrapper error", original)

        assert error.model == ModelName.OPENAI_GPT4O
        assert error.original_error == original
        assert str(error) == "gpt-4o: Wrapper error"

    def test_error_inheritance(self):
        """Test that LLMError inherits from Exception."""
        error = LLMError(ModelName.OPENAI_GPT4, "Test")
        assert isinstance(error, Exception)


class TestLLMModel:
    """Test LLMModel abstract base class."""

    def test_abstract_methods(self):
        """Test that LLMModel is properly abstract."""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            LLMModel(ModelName.OPENAI_GPT4)

    def test_concrete_implementation_required(self):
        """Test that concrete implementations must implement abstract methods."""

        class IncompleteModel(LLMModel):
            pass  # Missing all implementations

        # Should raise TypeError due to unimplemented abstract methods
        with pytest.raises(TypeError):
            IncompleteModel(ModelName.OPENAI_GPT4)

    def test_base_functionality(self):
        """Test base functionality works in concrete implementation."""

        class TestModel(LLMModel):
            async def generate_text(self, system_prompt, user_prompt, **kwargs):
                return Mock()

            async def generate_structured(
                self, system_prompt, user_prompt, schema, **kwargs
            ):
                return Mock()

        model = TestModel(ModelName.OPENAI_GPT4, timeout_seconds=60)
        assert model.model_name == ModelName.OPENAI_GPT4
        assert model.timeout_seconds == 60
        assert model.get_model_name() == ModelName.OPENAI_GPT4


class TestStandardLLMModel:
    """Test StandardLLMModel abstract base class."""

    def test_abstract_methods(self):
        """Test that StandardLLMModel is properly abstract."""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            StandardLLMModel(ModelName.OPENAI_GPT4)

    def test_concrete_implementation_required(self):
        """Test that concrete implementations must implement abstract methods."""

        class IncompleteStandardModel(StandardLLMModel):
            pass  # Missing implementations

        # Should raise TypeError due to unimplemented abstract methods
        with pytest.raises(TypeError):
            IncompleteStandardModel(ModelName.OPENAI_GPT4)


class TestTwoStageLLMModel:
    """Test TwoStageLLMModel abstract base class."""

    def test_abstract_methods(self):
        """Test that TwoStageLLMModel is properly abstract."""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            TwoStageLLMModel(ModelName.OPENAI_O3_DEEP_RESEARCH)

    def test_concrete_implementation_required(self):
        """Test that concrete implementations must implement abstract methods."""

        class IncompleteTwoStageModel(TwoStageLLMModel):
            pass  # Missing implementations

        # Should raise TypeError due to unimplemented abstract methods
        with pytest.raises(TypeError):
            IncompleteTwoStageModel(ModelName.OPENAI_O3_DEEP_RESEARCH)


class TestLLMProvider:
    """Test LLMProvider abstract base class."""

    def test_abstract_methods(self):
        """Test that LLMProvider is properly abstract."""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            LLMProvider()

    def test_concrete_implementation_required(self):
        """Test that concrete implementations must implement abstract methods."""

        class IncompleteProvider(LLMProvider):
            pass  # Missing all implementations

        # Should raise TypeError due to unimplemented abstract methods
        with pytest.raises(TypeError):
            IncompleteProvider()


class TestOpenAIProvider:
    """Test OpenAIProvider implementation."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = OpenAIProvider()
        # OpenAI provider doesn't have a name attribute
        assert isinstance(provider, OpenAIProvider)

    def test_is_available_with_api_key(self):
        """Test availability check with API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()
            assert provider.is_available() is True

    def test_is_available_without_api_key(self):
        """Test availability check without API key."""
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider()
            assert provider.is_available() is False

    def test_get_supported_models(self):
        """Test getting supported models."""
        provider = OpenAIProvider()
        models = provider.get_supported_models()

        assert len(models) >= 4
        assert ModelName.OPENAI_GPT4 in models
        assert ModelName.OPENAI_GPT4O in models
        assert ModelName.OPENAI_O3_DEEP_RESEARCH in models
        assert ModelName.OPENAI_O4_MINI_DEEP_RESEARCH in models

    def test_create_model_standard(self):
        """Test creating standard model."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()
            model = provider.create_model(ModelName.OPENAI_GPT4, timeout_seconds=60)

            assert isinstance(model, OpenAIStandardModel)
            assert model.model_name == ModelName.OPENAI_GPT4
            assert model.timeout_seconds == 60

    def test_create_model_deep_research(self):
        """Test creating deep research model."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()
            model = provider.create_model(
                ModelName.OPENAI_O3_DEEP_RESEARCH, timeout_seconds=300
            )

            assert isinstance(model, OpenAITwoStageModel)
            assert model.model_name == ModelName.OPENAI_O3_DEEP_RESEARCH
            assert model.timeout_seconds == 300

    def test_create_model_unsupported(self):
        """Test creating unsupported model."""
        provider = OpenAIProvider()

        with pytest.raises(LLMError) as exc_info:
            provider.create_model(ModelName.CLAUDE_SONNET)

        assert "Unsupported OpenAI model" in str(exc_info.value)

    def test_create_model_without_api_key(self):
        """Test creating model without API key."""
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider()

            with pytest.raises(LLMError) as exc_info:
                provider.create_model(ModelName.OPENAI_GPT4)

            assert "Failed to initialize OpenAI client" in str(exc_info.value)


class TestClaudeProvider:
    """Test ClaudeProvider implementation."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = ClaudeProvider()
        # Claude provider doesn't have a name attribute
        assert isinstance(provider, ClaudeProvider)

    def test_is_available(self):
        """Test availability check (should be False - not implemented)."""
        provider = ClaudeProvider()
        assert provider.is_available() is False

    def test_get_supported_models(self):
        """Test getting supported models (should be empty - not implemented)."""
        provider = ClaudeProvider()
        models = provider.get_supported_models()
        assert len(models) == 0

    def test_create_model_not_implemented(self):
        """Test creating model (should fail - not implemented)."""
        provider = ClaudeProvider()

        with pytest.raises(LLMError) as exc_info:
            provider.create_model(ModelName.CLAUDE_SONNET)

        assert "not yet implemented" in str(exc_info.value)


class TestGeminiProvider:
    """Test GeminiProvider implementation."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = GeminiProvider()
        # Gemini provider doesn't have a name attribute
        assert isinstance(provider, GeminiProvider)

    def test_is_available(self):
        """Test availability check (should be False - not implemented)."""
        provider = GeminiProvider()
        assert provider.is_available() is False

    def test_get_supported_models(self):
        """Test getting supported models (should be empty - not implemented)."""
        provider = GeminiProvider()
        models = provider.get_supported_models()
        assert len(models) == 0

    def test_create_model_not_implemented(self):
        """Test creating model (should fail - not implemented)."""
        provider = GeminiProvider()

        with pytest.raises(LLMError) as exc_info:
            provider.create_model(ModelName.GEMINI_PRO)

        assert "not yet implemented" in str(exc_info.value)


class TestOpenAIStandardModel:
    """Test OpenAIStandardModel implementation."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client."""
        return Mock()

    def test_model_initialization(self, mock_openai_client):
        """Test model initialization."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.AsyncOpenAI", return_value=mock_openai_client),
        ):
            model = OpenAIStandardModel(ModelName.OPENAI_GPT4, timeout_seconds=60)

            assert model.model_name == ModelName.OPENAI_GPT4
            assert model.timeout_seconds == 60
            assert (
                model.api_model == "gpt-4o-mini"
            )  # GPT-4 maps to mini for better availability
            assert model.client == mock_openai_client

    def test_model_name_mapping(self):
        """Test model name mapping."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.AsyncOpenAI"),
        ):
            # Test various model mappings
            model_gpt4 = OpenAIStandardModel(ModelName.OPENAI_GPT4)
            assert model_gpt4.api_model == "gpt-4o-mini"  # GPT-4 maps to mini

            model_gpt4o = OpenAIStandardModel(ModelName.OPENAI_GPT4O)
            assert model_gpt4o.api_model == "gpt-4o"

            model_structured = OpenAIStandardModel(ModelName.OPENAI_GPT4O_STRUCTURED)
            assert model_structured.api_model == "gpt-4o-2024-08-06"

    @pytest.mark.asyncio
    async def test_generate_text_success(self, mock_openai_client):
        """Test successful text generation."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.AsyncOpenAI", return_value=mock_openai_client),
        ):
            # Setup mock response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Generated text response"
            mock_response.usage.total_tokens = 150

            mock_openai_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            model = OpenAIStandardModel(ModelName.OPENAI_GPT4, timeout_seconds=60)

            result = await model.generate_text(
                system_prompt="System prompt",
                user_prompt="User prompt",
                temperature=0.7,
                max_tokens=1000,
            )

            # Verify response
            assert isinstance(result, LLMResponse)
            assert result.content == "Generated text response"
            assert result.model == ModelName.OPENAI_GPT4
            assert result.latency_ms > 0
            assert "usage" in result.metadata

            # Verify API call
            mock_openai_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o-mini"  # GPT-4 maps to mini
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_completion_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_generate_structured_success(self, mock_openai_client):
        """Test successful structured generation."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.AsyncOpenAI", return_value=mock_openai_client),
        ):
            # Setup mock response for structured outputs
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[
                0
            ].message.content = '{"rating": 4, "summary": "Test"}'
            mock_response.usage.total_tokens = 200

            mock_openai_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            model = OpenAIStandardModel(
                ModelName.OPENAI_GPT4O_STRUCTURED, timeout_seconds=60
            )

            schema = {"type": "object", "properties": {"rating": {"type": "integer"}}}
            result = await model.generate_structured(
                system_prompt="System prompt",
                user_prompt="User prompt",
                schema=schema,
                temperature=0.1,
                max_tokens=2000,
            )

            # Verify response
            assert isinstance(result, LLMResponse)
            assert result.content == {"rating": 4, "summary": "Test"}
            assert result.model == ModelName.OPENAI_GPT4O_STRUCTURED
            assert result.latency_ms > 0
            assert "structured_output_used" in result.metadata

    @pytest.mark.asyncio
    async def test_generate_text_api_error(self, mock_openai_client):
        """Test text generation with API error."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.AsyncOpenAI", return_value=mock_openai_client),
        ):
            # Setup mock to raise exception
            mock_openai_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error")
            )

            model = OpenAIStandardModel(ModelName.OPENAI_GPT4, timeout_seconds=60)

            with pytest.raises(LLMError) as exc_info:
                await model.generate_text("System", "User")

            assert "Text generation failed" in str(exc_info.value)
            assert exc_info.value.model == ModelName.OPENAI_GPT4


class TestOpenAITwoStageModel:
    """Test OpenAITwoStageModel implementation."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client."""
        return Mock()

    def test_model_initialization(self, mock_openai_client):
        """Test two-stage model initialization."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.AsyncOpenAI", return_value=mock_openai_client),
        ):
            model = OpenAITwoStageModel(
                ModelName.OPENAI_O3_DEEP_RESEARCH, timeout_seconds=300
            )

            assert model.model_name == ModelName.OPENAI_O3_DEEP_RESEARCH
            assert model.timeout_seconds == 300
            assert model.reasoning_model == "gpt-4o"  # o3 maps to gpt-4o
            assert model.structured_model == "gpt-4o-2024-08-06"

    def test_model_name_mapping(self):
        """Test two-stage model name mapping."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.AsyncOpenAI"),
        ):
            # Test o3 mapping
            model_o3 = OpenAITwoStageModel(ModelName.OPENAI_O3_DEEP_RESEARCH)
            assert model_o3.reasoning_model == "gpt-4o"
            assert model_o3.structured_model == "gpt-4o-2024-08-06"

            # Test o4-mini mapping
            model_o4 = OpenAITwoStageModel(ModelName.OPENAI_O4_MINI_DEEP_RESEARCH)
            assert model_o4.reasoning_model == "gpt-4o-mini"
            assert model_o4.structured_model == "gpt-4o-2024-08-06"

    @pytest.mark.asyncio
    async def test_generate_enhanced_reasoning(self, mock_openai_client):
        """Test enhanced reasoning generation."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.AsyncOpenAI", return_value=mock_openai_client),
        ):
            # Setup mock response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Enhanced reasoning output"
            mock_response.usage.total_tokens = 300

            mock_openai_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            model = OpenAITwoStageModel(
                ModelName.OPENAI_O3_DEEP_RESEARCH, timeout_seconds=300
            )

            result = await model.generate_enhanced_reasoning(
                system_prompt="System prompt",
                user_prompt="User prompt",
                temperature=0.3,
                max_tokens=16000,
            )

            # Verify response
            assert isinstance(result, LLMResponse)
            assert result.content == "Enhanced reasoning output"
            assert result.model == ModelName.OPENAI_O3_DEEP_RESEARCH

            # Verify API call used reasoning model
            call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o"  # o3 maps to gpt-4o


class TestModelRegistry:
    """Test ModelRegistry class."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = ModelRegistry()

        # Should have all providers registered
        assert len(registry.providers) == 3
        assert "openai" in registry.providers
        assert "claude" in registry.providers
        assert "gemini" in registry.providers

    def test_get_all_supported_models(self):
        """Test getting all supported models."""
        registry = ModelRegistry()
        models = registry.get_all_supported_models()

        # Should include models from all providers
        assert len(models) >= 4  # At least OpenAI models
        assert ModelName.OPENAI_GPT4 in models
        assert ModelName.OPENAI_O3_DEEP_RESEARCH in models

    def test_get_available_models_with_api_key(self):
        """Test getting available models with API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            registry = ModelRegistry()
            models = registry.get_available_models()

            # Should include OpenAI models
            assert len(models) >= 4
            assert ModelName.OPENAI_GPT4 in models

    def test_get_available_models_without_api_key(self):
        """Test getting available models without API key."""
        with patch.dict(os.environ, {}, clear=True):
            registry = ModelRegistry()
            models = registry.get_available_models()

            # Should be empty without API keys
            assert len(models) == 0

    def test_create_model_success(self):
        """Test successful model creation."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            registry = ModelRegistry()
            model = registry.create_model(ModelName.OPENAI_GPT4, timeout_seconds=60)

            assert isinstance(model, OpenAIStandardModel)
            assert model.model_name == ModelName.OPENAI_GPT4

    def test_create_model_unsupported(self):
        """Test creating unsupported model."""
        registry = ModelRegistry()

        with pytest.raises(LLMError) as exc_info:
            registry.create_model(ModelName.CLAUDE_SONNET)

        assert "No provider found for model" in str(exc_info.value)

    def test_get_default_model_with_available(self):
        """Test getting default model when models are available."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            registry = ModelRegistry()
            default_model = registry.get_default_model()

            # Should prefer deep research models
            assert default_model in [
                ModelName.OPENAI_O3_DEEP_RESEARCH,
                ModelName.OPENAI_O4_MINI_DEEP_RESEARCH,
            ]

    def test_get_default_model_no_available(self):
        """Test getting default model when no models are available."""
        with patch.dict(os.environ, {}, clear=True):
            registry = ModelRegistry()

            with pytest.raises(LLMError) as exc_info:
                registry.get_default_model()

            assert "No models available" in str(exc_info.value)

    def test_create_default_model_success(self):
        """Test successful default model creation."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            registry = ModelRegistry()
            model = registry.create_default_model(timeout_seconds=120)

            # Should create a model instance
            assert isinstance(model, (OpenAIStandardModel, OpenAITwoStageModel))
            assert model.timeout_seconds == 120


class TestLLMModelFactory:
    """Test LLMModelFactory class."""

    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = LLMModelFactory()
        assert isinstance(factory.registry, ModelRegistry)

    def test_get_supported_models(self):
        """Test getting supported models through factory."""
        factory = LLMModelFactory()
        models = factory.get_supported_models()

        assert len(models) >= 4
        assert ModelName.OPENAI_GPT4 in models

    def test_get_available_models_with_api_key(self):
        """Test getting available models with API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            factory = LLMModelFactory()
            models = factory.get_available_models()

            assert len(models) >= 4
            assert ModelName.OPENAI_GPT4 in models

    def test_create_model_success(self):
        """Test successful model creation through factory."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            factory = LLMModelFactory()
            model = factory.create_model(ModelName.OPENAI_GPT4, timeout_seconds=60)

            assert isinstance(model, OpenAIStandardModel)
            assert model.model_name == ModelName.OPENAI_GPT4

    def test_create_default_model_success(self):
        """Test successful default model creation through factory."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            factory = LLMModelFactory()
            model = factory.create_default_model(timeout_seconds=120)

            assert isinstance(model, (OpenAIStandardModel, OpenAITwoStageModel))
            assert model.timeout_seconds == 120


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_create_model_function(self):
        """Test global create_model function."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            model = create_model(ModelName.OPENAI_GPT4, timeout_seconds=60)

            assert isinstance(model, OpenAIStandardModel)
            assert model.model_name == ModelName.OPENAI_GPT4

    def test_get_supported_models_function(self):
        """Test global get_supported_models function."""
        models = get_supported_models()

        assert len(models) >= 4
        assert ModelName.OPENAI_GPT4 in models

    def test_get_available_models_function(self):
        """Test global get_available_models function."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            models = get_available_models()

            assert len(models) >= 4
            assert ModelName.OPENAI_GPT4 in models

    def test_create_default_model_function(self):
        """Test global create_default_model function."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            model = create_default_model(timeout_seconds=120)

            assert isinstance(model, (OpenAIStandardModel, OpenAITwoStageModel))
            assert model.timeout_seconds == 120


class TestModelMapping:
    """Test model name mapping functionality."""

    def test_openai_standard_model_mappings(self):
        """Test OpenAI standard model API name mappings."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.AsyncOpenAI"),
        ):
            test_cases = [
                (ModelName.OPENAI_GPT4, "gpt-4o-mini"),  # GPT-4 maps to mini
                (ModelName.OPENAI_GPT4O, "gpt-4o"),
                (ModelName.OPENAI_GPT4O_STRUCTURED, "gpt-4o-2024-08-06"),
            ]

            for model_name, expected_api_name in test_cases:
                model = OpenAIStandardModel(model_name)
                assert model.api_model == expected_api_name

    def test_openai_two_stage_model_mappings(self):
        """Test OpenAI two-stage model API name mappings."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.AsyncOpenAI"),
        ):
            # Test o3 two-stage mapping
            model_o3 = OpenAITwoStageModel(ModelName.OPENAI_O3_DEEP_RESEARCH)
            assert model_o3.reasoning_model == "gpt-4o"
            assert model_o3.structured_model == "gpt-4o-2024-08-06"

            # Test o4-mini two-stage mapping
            model_o4 = OpenAITwoStageModel(ModelName.OPENAI_O4_MINI_DEEP_RESEARCH)
            assert model_o4.reasoning_model == "gpt-4o-mini"
            assert model_o4.structured_model == "gpt-4o-2024-08-06"


class TestErrorHandling:
    """Test error handling across the module."""

    def test_openai_client_initialization_failure(self):
        """Test OpenAI client initialization failure."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMError) as exc_info:
                OpenAIStandardModel(ModelName.OPENAI_GPT4)

            assert "Failed to initialize OpenAI client" in str(exc_info.value)
            assert exc_info.value.model == ModelName.OPENAI_GPT4

    def test_unsupported_model_error_handling(self):
        """Test error handling for unsupported model operations."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(LLMError) as exc_info:
                OpenAIStandardModel(ModelName.CLAUDE_SONNET)

            assert "Failed to initialize OpenAI client" in str(exc_info.value)
            assert exc_info.value.model == ModelName.CLAUDE_SONNET

    def test_timeout_configuration(self):
        """Test timeout configuration in models."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.AsyncOpenAI"),
        ):
            # Test default timeout
            model_default = OpenAIStandardModel(ModelName.OPENAI_GPT4)
            assert model_default.timeout_seconds == 300

            # Test custom timeout
            model_custom = OpenAIStandardModel(
                ModelName.OPENAI_GPT4, timeout_seconds=120
            )
            assert model_custom.timeout_seconds == 120
