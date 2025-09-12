"""Pure LLM abstraction layer for managing different providers and models.

This module provides a clean abstraction for LLM providers without any
domain-specific logic. It handles provider implementations, model creation,
and basic API interactions.

Architecture:
- LLMModel: Base interface for all models
- StandardLLMModel: Single-stage models for direct responses
- TwoStageLLMModel: Two-stage models with enhanced reasoning
- LLMProvider: Provider-specific implementations
"""

import abc
import json
import time
import re
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from enum import Enum

from dotenv import load_dotenv


# Load environment variables with override
load_dotenv(override=True)


class ModelName(Enum):
    """Supported models for sector analysis.

    Includes both standard single-stage models and deep research models
    with two-stage pipelines across multiple providers.
    """

    # OpenAI Deep Research Models
    OPENAI_O3_DEEP_RESEARCH = "o3-deep-research"
    OPENAI_O4_MINI_DEEP_RESEARCH = "o4-mini-deep-research"

    # OpenAI Standard Models
    OPENAI_GPT4 = "gpt-4"
    OPENAI_GPT4O = "gpt-4o"
    OPENAI_GPT4O_STRUCTURED = "gpt-4o-2024-08-06"

    # Claude Models (future)
    CLAUDE_SONNET = "claude-3-sonnet"
    CLAUDE_HAIKU = "claude-3-haiku"

    # Gemini Models (future)
    GEMINI_PRO = "gemini-pro"
    GEMINI_FLASH = "gemini-1.5-flash"


class LLMError(Exception):
    """Base exception for LLM-related errors.

    Attributes:
        model: The model that failed.
        message: Description of the error.
        original_error: The underlying exception (if any).
    """

    def __init__(
        self, model: ModelName, message: str, original_error: Optional[Exception] = None
    ):
        self.model = model
        self.original_error = original_error
        super().__init__(f"{model.value}: {message}")


class LLMResponse:
    """Standard response format for LLM operations.

    Attributes:
        content: The response content (text or structured data).
        model: The model that generated the response.
        latency_ms: Response time in milliseconds.
        timestamp_utc: When the response was generated.
        metadata: Additional provider-specific metadata.
    """

    def __init__(
        self,
        content: Union[str, Dict[str, Any]],
        model: ModelName,
        latency_ms: float,
        timestamp_utc: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content = content
        self.model = model
        self.latency_ms = latency_ms
        self.timestamp_utc = timestamp_utc
        self.metadata = metadata or {}


class LLMModel(abc.ABC):
    """Abstract base class for all LLM models.

    This defines the core interface that all model implementations must follow,
    ensuring consistent behavior across different providers and model types.
    """

    def __init__(self, model_name: ModelName, timeout_seconds: int = 300):
        """Initialize the LLM model.

        Args:
            model_name: The model identifier.
            timeout_seconds: Maximum time to wait for responses.
        """
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    @abc.abstractmethod
    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        """Generate text response from prompts.

        Args:
            system_prompt: System instructions for the model.
            user_prompt: User prompt or query.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse containing the generated text.

        Raises:
            LLMError: If generation fails or times out.
        """
        pass

    @abc.abstractmethod
    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        """Generate structured JSON response from prompts.

        Args:
            system_prompt: System instructions for the model.
            user_prompt: User prompt or query.
            schema: JSON schema for structured output.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with structured data in content field.

        Raises:
            LLMError: If generation fails or times out.
        """
        pass

    def get_model_name(self) -> ModelName:
        """Get the model name for this instance.

        Returns:
            ModelName enum value.
        """
        return self.model_name


class StandardLLMModel(LLMModel):
    """Base class for single-stage LLM models.

    Standard models use a single API call to generate responses directly.
    This is suitable for most modern LLMs with good instruction following.
    """

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        """Generate structured response using single-stage approach.

        Default implementation tries to generate structured output directly.
        Subclasses can override for provider-specific structured output support.

        Args:
            system_prompt: System instructions.
            user_prompt: User prompt.
            schema: JSON schema for output.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            LLMResponse with structured data.

        Raises:
            LLMError: If generation fails.
        """
        # Add schema instructions to system prompt
        enhanced_system_prompt = (
            f"{system_prompt}\n\n"
            f"IMPORTANT: Respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )

        # Generate text response
        text_response = await self.generate_text(
            enhanced_system_prompt, user_prompt, temperature, max_tokens
        )

        # Parse JSON from text response
        try:
            structured_data = self._extract_json_from_text(text_response.content)
            return LLMResponse(
                content=structured_data,
                model=self.model_name,
                latency_ms=text_response.latency_ms,
                timestamp_utc=text_response.timestamp_utc,
                metadata=text_response.metadata,
            )
        except Exception as e:
            raise LLMError(
                self.model_name, f"Failed to parse structured output: {str(e)}", e
            )

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text response.

        Args:
            text: Text response that may contain JSON.

        Returns:
            Parsed JSON data.

        Raises:
            json.JSONDecodeError: If JSON cannot be extracted.
        """
        # Try parsing the entire response as JSON first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in text
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        raise json.JSONDecodeError("No valid JSON found in response", text, 0)


class TwoStageLLMModel(LLMModel):
    """Base class for two-stage models with enhanced reasoning.

    Two-stage models use enhanced reasoning capabilities in the first stage
    for comprehensive analysis, then convert to structured output in the
    second stage for perfect compliance.
    """

    @abc.abstractmethod
    async def generate_enhanced_reasoning(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        """Generate response using enhanced reasoning model.

        Args:
            system_prompt: System instructions.
            user_prompt: User prompt for reasoning.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens for reasoning.

        Returns:
            LLLResponse with comprehensive reasoning content.

        Raises:
            LLMError: If enhanced reasoning fails.
        """
        pass

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        """Generate structured response using two-stage approach.

        Stage 1: Use enhanced reasoning for comprehensive analysis
        Stage 2: Convert to structured output with high compliance

        Args:
            system_prompt: System instructions.
            user_prompt: User prompt.
            schema: JSON schema for structured output.
            temperature: Sampling temperature for stage 2.
            max_tokens: Maximum tokens for stage 2.

        Returns:
            LLMResponse with structured data.

        Raises:
            LLMError: If two-stage generation fails.
        """
        start_time = time.time()

        try:
            # Stage 1: Enhanced reasoning
            reasoning_response = await self.generate_enhanced_reasoning(
                system_prompt, user_prompt, temperature=0.3, max_tokens=16000
            )

            # Stage 2: Convert to structured output
            conversion_prompt = (
                f"Convert the following analysis into structured JSON format "
                f"matching the required schema:\n\n{reasoning_response.content}"
            )

            structured_response = await self._convert_to_structured(
                conversion_prompt, schema, temperature, max_tokens
            )

            # Calculate total latency
            total_latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=structured_response.content,
                model=self.model_name,
                latency_ms=total_latency_ms,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                metadata={
                    "stage1_latency_ms": reasoning_response.latency_ms,
                    "stage2_latency_ms": structured_response.latency_ms,
                    "reasoning_content_length": len(reasoning_response.content),
                },
            )

        except Exception as e:
            raise LLMError(self.model_name, f"Two-stage generation failed: {str(e)}", e)

    @abc.abstractmethod
    async def _convert_to_structured(
        self,
        conversion_prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        """Convert content to structured output using structured output model.

        Args:
            conversion_prompt: Prompt with content to convert.
            schema: JSON schema for output.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            LLMResponse with structured data.

        Raises:
            LLMError: If conversion fails.
        """
        pass


class OpenAIStandardModel(StandardLLMModel):
    """OpenAI implementation for standard models.

    Single-stage implementation for standard OpenAI models like GPT-4, GPT-4o, etc.
    """

    def __init__(self, model_name: ModelName, timeout_seconds: int = 300):
        """Initialize the OpenAI standard model.

        Args:
            model_name: The standard model to use.
            timeout_seconds: Maximum time to wait for analysis.

        Raises:
            LLMError: If initialization fails.
        """
        super().__init__(model_name, timeout_seconds)

        # Import OpenAI client (lazy import for optional dependency)
        try:
            from openai import AsyncOpenAI
            import os

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMError(
                    model_name, "OPENAI_API_KEY environment variable not set"
                )

            self.client = AsyncOpenAI(api_key=api_key)

            # Map logical model names to actual API model names
            self.model_mapping = {
                ModelName.OPENAI_GPT4: "gpt-4o-mini",  # Use mini for better availability
                ModelName.OPENAI_GPT4O: "gpt-4o",
                ModelName.OPENAI_GPT4O_STRUCTURED: "gpt-4o-2024-08-06",
            }

            # Get the actual model name for API calls
            if model_name not in self.model_mapping:
                raise LLMError(model_name, f"Unsupported standard model: {model_name}")

            self.api_model = self.model_mapping[model_name]

        except ImportError as e:
            raise LLMError(
                model_name,
                "OpenAI package not available. Install with: uv add openai",
                e,
            )
        except Exception as e:
            raise LLMError(model_name, "Failed to initialize OpenAI client", e)

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        """Generate text response using OpenAI API.

        Args:
            system_prompt: System instructions.
            user_prompt: User prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            LLMResponse with text content.

        Raises:
            LLMError: If generation fails.
        """
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.api_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMError(self.model_name, "Empty response from OpenAI")

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=content,
                model=self.model_name,
                latency_ms=latency_ms,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                metadata={
                    "api_model": self.api_model,
                    "usage": response.usage.model_dump() if response.usage else {},
                },
            )

        except Exception as e:
            raise LLMError(self.model_name, f"Text generation failed: {str(e)}", e)

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        """Generate structured response using OpenAI structured outputs.

        Args:
            system_prompt: System instructions.
            user_prompt: User prompt.
            schema: JSON schema for output.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            LLMResponse with structured data.

        Raises:
            LLMError: If generation fails.
        """
        start_time = time.time()

        try:
            # Use structured outputs if available (gpt-4o-2024-08-06)
            if self.api_model == "gpt-4o-2024-08-06":
                response = await self.client.chat.completions.create(
                    model=self.api_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_response",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )

                content = response.choices[0].message.content
                if not content:
                    raise LLMError(self.model_name, "Empty structured response")

                structured_data = json.loads(content)

            else:
                # Fallback to text generation with JSON parsing
                text_response = await self.generate_text(
                    system_prompt, user_prompt, temperature, max_tokens
                )
                structured_data = self._extract_json_from_text(text_response.content)

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=structured_data,
                model=self.model_name,
                latency_ms=latency_ms,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                metadata={
                    "api_model": self.api_model,
                    "structured_output_used": self.api_model == "gpt-4o-2024-08-06",
                },
            )

        except Exception as e:
            raise LLMError(
                self.model_name, f"Structured generation failed: {str(e)}", e
            )


class OpenAITwoStageModel(TwoStageLLMModel):
    """OpenAI implementation for two-stage models.

    Uses the actual deep research models (o3-deep-research, o4-mini-deep-research)
    for enhanced reasoning, followed by gpt-4o-2024-08-06 for structured output.
    """

    def __init__(self, model_name: ModelName, timeout_seconds: int = 300):
        """Initialize the OpenAI two-stage model.

        Args:
            model_name: The two-stage model to use.
            timeout_seconds: Maximum time to wait for analysis.

        Raises:
            LLMError: If initialization fails.
        """
        super().__init__(model_name, timeout_seconds)

        # Import OpenAI client (lazy import for optional dependency)
        try:
            from openai import AsyncOpenAI
            import os

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMError(
                    model_name, "OPENAI_API_KEY environment variable not set"
                )

            self.client = AsyncOpenAI(api_key=api_key)

            # Map logical model names to actual API model names
            # Note: Using GPT-4o as fallback until actual deep research models are available
            self.reasoning_model_mapping = {
                ModelName.OPENAI_O3_DEEP_RESEARCH: "gpt-4o",  # Fallback until o3-deep-research available
                ModelName.OPENAI_O4_MINI_DEEP_RESEARCH: "gpt-4o-mini",  # Fallback until o4-mini-deep-research available
            }

            # Get the actual model name for API calls
            if model_name not in self.reasoning_model_mapping:
                raise LLMError(model_name, f"Unsupported two-stage model: {model_name}")

            self.reasoning_model = self.reasoning_model_mapping[model_name]
            self.structured_model = "gpt-4o-2024-08-06"

        except ImportError as e:
            raise LLMError(
                model_name,
                "OpenAI package not available. Install with: uv add openai",
                e,
            )
        except Exception as e:
            raise LLMError(model_name, "Failed to initialize OpenAI client", e)

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        """Generate text using reasoning model.

        Args:
            system_prompt: System instructions.
            user_prompt: User prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            LLMResponse with text content.

        Raises:
            LLMError: If generation fails.
        """
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.reasoning_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMError(self.model_name, "Empty response")

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=content,
                model=self.model_name,
                latency_ms=latency_ms,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                metadata={
                    "api_model": self.reasoning_model,
                    "stage": "reasoning",
                },
            )

        except Exception as e:
            raise LLMError(self.model_name, f"Enhanced reasoning failed: {str(e)}", e)

    async def generate_enhanced_reasoning(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        """Generate enhanced reasoning response.

        Args:
            system_prompt: System instructions.
            user_prompt: User prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            LLMResponse with reasoning content.

        Raises:
            LLMError: If reasoning fails.
        """
        return await self.generate_text(
            system_prompt, user_prompt, temperature, max_tokens
        )

    async def _convert_to_structured(
        self,
        conversion_prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        """Convert content to structured output using gpt-4o-2024-08-06.

        Args:
            conversion_prompt: Prompt with content to convert.
            schema: JSON schema for output.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            LLMResponse with structured data.

        Raises:
            LLMError: If conversion fails.
        """
        start_time = time.time()

        try:
            # Use gpt-4o-2024-08-06 with structured outputs
            response = await self.client.chat.completions.create(
                model=self.structured_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data conversion specialist. Convert the provided content into the exact JSON format specified.",
                    },
                    {"role": "user", "content": conversion_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "converted_data",
                        "schema": schema,
                        "strict": True,
                    },
                },
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMError(self.model_name, "Empty structured output")

            structured_data = json.loads(content)
            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=structured_data,
                model=self.model_name,
                latency_ms=latency_ms,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                metadata={
                    "api_model": self.structured_model,
                    "stage": "structured_conversion",
                },
            )

        except Exception as e:
            raise LLMError(
                self.model_name, f"Structured conversion failed: {str(e)}", e
            )


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers.

    Providers handle the creation and configuration of models from their
    respective APIs (OpenAI, Anthropic, Google, etc.).
    """

    @abc.abstractmethod
    def get_supported_models(self) -> List[ModelName]:
        """Get list of models supported by this provider.

        Returns:
            List of supported ModelName values.
        """
        pass

    @abc.abstractmethod
    def create_model(
        self, model_name: ModelName, timeout_seconds: int = 300
    ) -> LLMModel:
        """Create a model instance for the specified model.

        Args:
            model_name: The model to create.
            timeout_seconds: Timeout configuration.

        Returns:
            Configured LLMModel instance.

        Raises:
            LLMError: If model creation fails.
        """
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (API key configured, etc.).

        Returns:
            True if provider is available, False otherwise.
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation.

    Manages all OpenAI models including both standard and two-stage variants.
    """

    def get_supported_models(self) -> List[ModelName]:
        """Get list of models supported by OpenAI provider.

        Returns:
            List of supported ModelName values.
        """
        return [
            # Two-Stage Models (Deep Research)
            ModelName.OPENAI_O3_DEEP_RESEARCH,
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH,
            # Standard Models
            ModelName.OPENAI_GPT4,
            ModelName.OPENAI_GPT4O,
            ModelName.OPENAI_GPT4O_STRUCTURED,
        ]

    def create_model(
        self, model_name: ModelName, timeout_seconds: int = 300
    ) -> LLMModel:
        """Create an OpenAI model instance.

        Args:
            model_name: The model to create.
            timeout_seconds: Timeout configuration.

        Returns:
            Configured LLMModel instance.

        Raises:
            LLMError: If model creation fails.
        """
        # Two-stage models (deep research)
        if model_name in [
            ModelName.OPENAI_O3_DEEP_RESEARCH,
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH,
        ]:
            return OpenAITwoStageModel(model_name, timeout_seconds)

        # Standard models
        elif model_name in [
            ModelName.OPENAI_GPT4,
            ModelName.OPENAI_GPT4O,
            ModelName.OPENAI_GPT4O_STRUCTURED,
        ]:
            return OpenAIStandardModel(model_name, timeout_seconds)

        else:
            raise LLMError(model_name, f"Unsupported OpenAI model: {model_name}")

    def is_available(self) -> bool:
        """Check if OpenAI provider is available.

        Returns:
            True if OPENAI_API_KEY is configured, False otherwise.
        """
        import os

        return bool(os.getenv("OPENAI_API_KEY"))


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider implementation.

    Placeholder implementation for future Claude support.
    """

    def get_supported_models(self) -> List[ModelName]:
        """Get list of models supported by Claude provider.

        Returns:
            Empty list - not implemented yet.
        """
        return []

    def create_model(
        self, model_name: ModelName, timeout_seconds: int = 300
    ) -> LLMModel:
        """Create a Claude model instance.

        Args:
            model_name: The model to create.
            timeout_seconds: Timeout configuration.

        Returns:
            Configured LLMModel instance.

        Raises:
            LLMError: Always raises - not implemented yet.
        """
        raise LLMError(model_name, "Claude provider not yet implemented")

    def is_available(self) -> bool:
        """Check if Claude provider is available.

        Returns:
            False - not implemented yet.
        """
        return False


class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation.

    Placeholder implementation for future Gemini support.
    """

    def get_supported_models(self) -> List[ModelName]:
        """Get list of models supported by Gemini provider.

        Returns:
            Empty list - not implemented yet.
        """
        return []

    def create_model(
        self, model_name: ModelName, timeout_seconds: int = 300
    ) -> LLMModel:
        """Create a Gemini model instance.

        Args:
            model_name: The model to create.
            timeout_seconds: Timeout configuration.

        Returns:
            Configured LLMModel instance.

        Raises:
            LLMError: Always raises - not implemented yet.
        """
        raise LLMError(model_name, "Gemini provider not yet implemented")

    def is_available(self) -> bool:
        """Check if Gemini provider is available.

        Returns:
            False - not implemented yet.
        """
        return False


class ModelRegistry:
    """Central registry for all available LLM models and providers.

    This class manages the registration and discovery of models across all
    providers, providing a unified interface for model creation.
    """

    def __init__(self):
        """Initialize the model registry."""
        self.providers: Dict[str, LLMProvider] = {}
        self._register_default_providers()

    def _register_default_providers(self) -> None:
        """Register default providers."""
        self.providers["openai"] = OpenAIProvider()
        self.providers["claude"] = ClaudeProvider()
        self.providers["gemini"] = GeminiProvider()

    def get_all_supported_models(self) -> List[ModelName]:
        """Get all models supported by any provider.

        Returns:
            List of all supported ModelName values.
        """
        all_models = []
        for provider in self.providers.values():
            all_models.extend(provider.get_supported_models())
        return all_models

    def get_available_models(self) -> List[ModelName]:
        """Get models from available providers (with API keys configured).

        Returns:
            List of ModelName values from available providers.
        """
        available_models = []
        for provider in self.providers.values():
            if provider.is_available():
                available_models.extend(provider.get_supported_models())
        return available_models

    def create_model(
        self, model_name: ModelName, timeout_seconds: int = 300
    ) -> LLMModel:
        """Create a model instance from any available provider.

        Args:
            model_name: The model to create.
            timeout_seconds: Timeout configuration.

        Returns:
            Configured LLMModel instance.

        Raises:
            LLMError: If no provider supports the model or creation fails.
        """
        for provider in self.providers.values():
            if model_name in provider.get_supported_models():
                if provider.is_available():
                    return provider.create_model(model_name, timeout_seconds)
                else:
                    raise LLMError(
                        model_name,
                        f"No provider available for {model_name} (missing API key?)",
                    )

        raise LLMError(model_name, f"No provider found for model: {model_name}")

    def get_default_model(self) -> ModelName:
        """Get the default model name.

        Uses the first available two-stage model, falling back to
        standard models if necessary.

        Returns:
            Default ModelName.

        Raises:
            LLMError: If no models are available.
        """
        available_models = self.get_available_models()
        if not available_models:
            raise LLMError(
                ModelName.OPENAI_O4_MINI_DEEP_RESEARCH,
                "No models available (check API keys)",
            )

        # Prefer two-stage models for enhanced reasoning
        two_stage_models = [
            ModelName.OPENAI_O3_DEEP_RESEARCH,
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH,
        ]

        for model in two_stage_models:
            if model in available_models:
                return model

        # Fallback to first available model
        return available_models[0]

    def create_default_model(self, timeout_seconds: int = 300) -> LLMModel:
        """Create the default model instance.

        Uses the first available two-stage model, falling back to
        standard models if necessary.

        Args:
            timeout_seconds: Timeout configuration.

        Returns:
            Default LLMModel instance.

        Raises:
            LLMError: If no models are available.
        """
        default_model_name = self.get_default_model()
        return self.create_model(default_model_name, timeout_seconds)


class LLMModelFactory:
    """Factory for creating LLM model instances.

    This factory provides a simple interface for creating models while
    handling provider discovery and configuration automatically.
    """

    def __init__(self):
        """Initialize the model factory."""
        self.registry = ModelRegistry()

    def create_model(
        self, model_name: ModelName, timeout_seconds: int = 300
    ) -> LLMModel:
        """Create a model instance for the specified model.

        Args:
            model_name: The model to create.
            timeout_seconds: Timeout configuration.

        Returns:
            Configured LLMModel instance.

        Raises:
            LLMError: If model creation fails.
        """
        return self.registry.create_model(model_name, timeout_seconds)

    def get_supported_models(self) -> List[ModelName]:
        """Get list of all supported models.

        Returns:
            List of supported ModelName values.
        """
        return self.registry.get_all_supported_models()

    def get_available_models(self) -> List[ModelName]:
        """Get list of models from available providers.

        Returns:
            List of available ModelName values.
        """
        return self.registry.get_available_models()

    def create_default_model(self, timeout_seconds: int = 300) -> LLMModel:
        """Create the default model instance.

        Uses the first available two-stage model, falling back to
        standard models if necessary.

        Args:
            timeout_seconds: Timeout configuration.

        Returns:
            Default LLMModel instance.

        Raises:
            LLMError: If no models are available.
        """
        available_models = self.get_available_models()
        if not available_models:
            raise LLMError(
                ModelName.OPENAI_O4_MINI_DEEP_RESEARCH,
                "No models available (check API keys)",
            )

        # Prefer two-stage models for enhanced reasoning
        two_stage_models = [
            ModelName.OPENAI_O3_DEEP_RESEARCH,
            ModelName.OPENAI_O4_MINI_DEEP_RESEARCH,
        ]

        for model in two_stage_models:
            if model in available_models:
                return self.create_model(model, timeout_seconds)

        # Fallback to first available model
        return self.create_model(available_models[0], timeout_seconds)


# Global factory instance for convenience
_model_factory = LLMModelFactory()


def create_model(model_name: ModelName, timeout_seconds: int = 300) -> LLMModel:
    """Create a model instance using the global factory.

    Args:
        model_name: The model to create.
        timeout_seconds: Timeout configuration.

    Returns:
        Configured LLMModel instance.

    Raises:
        LLMError: If model creation fails.
    """
    return _model_factory.create_model(model_name, timeout_seconds)


def get_supported_models() -> List[ModelName]:
    """Get list of all supported models.

    Returns:
        List of supported ModelName values.
    """
    return _model_factory.get_supported_models()


def get_available_models() -> List[ModelName]:
    """Get list of models from available providers.

    Returns:
        List of available ModelName values.
    """
    return _model_factory.get_available_models()


def create_default_model(timeout_seconds: int = 300) -> LLMModel:
    """Create the default model instance.

    Args:
        timeout_seconds: Timeout configuration.

    Returns:
        Default LLMModel instance.

    Raises:
        LLMError: If no models are available.
    """
    return _model_factory.create_default_model(timeout_seconds)
