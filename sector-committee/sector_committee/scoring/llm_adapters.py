"""Sector-specific LLM adapters that bridge pure LLM models with scoring logic.

This module creates sector analysis adapters on top of the generic LLM models,
adding domain-specific functionality like prompt building, validation, and
the analyze_sector interface.

Architecture:
- SectorAnalysisAdapter: Base adapter for sector analysis
- StandardSectorAdapter: Single-stage sector analysis
- DeepResearchSectorAdapter: Two-stage deep research sector analysis
"""

import abc
import asyncio
import time
import re
import aiohttp
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from ..data_models import SectorRequest, ModelResult
from ..llm_models import ModelName
from ..llm_models import (
    LLMModel,
    LLMError,
    StandardLLMModel,
    TwoStageLLMModel,
    create_model,
)
from .schema import validate_sector_rating, create_openai_structured_output_schema
from .prompts import (
    build_deep_research_system_prompt,
    build_deep_research_user_prompt,
    build_conversion_prompt,
    build_system_prompt,
    build_user_prompt,
    build_structured_conversion_system_prompt,
)


class SectorAnalysisError(Exception):
    """Exception raised when sector analysis fails.

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


class SectorAnalysisAdapter(abc.ABC):
    """Abstract base class for sector analysis adapters.

    This defines the interface for sector-specific analysis using
    underlying LLM models while adding domain-specific functionality.
    """

    def __init__(self, llm_model: LLMModel):
        """Initialize the sector analysis adapter.

        Args:
            llm_model: The underlying LLM model to use.
        """
        self.llm_model = llm_model

    @abc.abstractmethod
    async def analyze_sector(self, request: SectorRequest) -> ModelResult:
        """Analyze a sector and return structured results.

        Args:
            request: Sector analysis request with parameters.

        Returns:
            ModelResult containing the analysis and metadata.

        Raises:
            SectorAnalysisError: If analysis fails or times out.
        """
        pass

    def get_model_name(self) -> ModelName:
        """Get the model name for this adapter.

        Returns:
            ModelName enum value.
        """
        return self.llm_model.get_model_name()


class StandardSectorAdapter(SectorAnalysisAdapter):
    """Sector analysis adapter for single-stage LLM models.

    Uses standard models to generate structured sector analysis directly
    in a single API call.
    """

    def __init__(self, llm_model: StandardLLMModel):
        """Initialize the standard sector adapter.

        Args:
            llm_model: The standard LLM model to use.
        """
        super().__init__(llm_model)

    async def analyze_sector(self, request: SectorRequest) -> ModelResult:
        """Analyze sector using single-stage approach.

        Args:
            request: Sector analysis request.

        Returns:
            ModelResult with analysis results.

        Raises:
            SectorAnalysisError: If analysis fails or times out.
        """
        start_time = time.time()
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        try:
            # Build sector-specific prompts
            system_prompt = build_system_prompt()
            user_prompt = build_user_prompt(request)
            schema = create_openai_structured_output_schema()

            # Generate structured response using the LLM model
            llm_response = await self.llm_model.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
                temperature=0.1,
                max_tokens=8000,
            )

            # Validate the response
            response_data = llm_response.content
            validation_result = validate_sector_rating(response_data)
            if not validation_result.is_valid:
                raise SectorAnalysisError(
                    self.llm_model.model_name,
                    f"Invalid response format: {validation_result.errors}",
                )

            # Calculate total latency
            latency_ms = (time.time() - start_time) * 1000

            return ModelResult(
                model=self.llm_model.model_name,
                data=response_data,  # type: ignore (validated as SectorRating)
                latency_ms=latency_ms,
                timestamp_utc=timestamp_utc,
                success=True,
            )

        except asyncio.TimeoutError:
            latency_ms = self.llm_model.timeout_seconds * 1000
            raise SectorAnalysisError(
                self.llm_model.model_name,
                f"Analysis timed out after {self.llm_model.timeout_seconds} seconds",
            )
        except LLMError as e:
            latency_ms = (time.time() - start_time) * 1000
            raise SectorAnalysisError(e.model, f"LLM error: {str(e)}", e)
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            raise SectorAnalysisError(
                self.llm_model.model_name, f"Analysis failed: {str(e)}", e
            )


class DeepResearchSectorAdapter(SectorAnalysisAdapter):
    """Sector analysis adapter for two-stage deep research models.

    Uses enhanced reasoning capabilities in the first stage for comprehensive
    sector research, then converts to structured output in the second stage.
    """

    def __init__(self, llm_model: TwoStageLLMModel):
        """Initialize the deep research sector adapter.

        Args:
            llm_model: The two-stage LLM model to use.
        """
        super().__init__(llm_model)

    async def analyze_sector(self, request: SectorRequest) -> ModelResult:
        """Analyze sector using two-stage deep research pipeline.

        Stage 1: Conduct comprehensive research with enhanced reasoning
        Stage 2: Convert to structured JSON output

        Args:
            request: Sector analysis request.

        Returns:
            ModelResult with structured sector rating.

        Raises:
            SectorAnalysisError: If analysis fails or times out.
        """
        start_time = time.time()
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        try:
            # Stage 1: Deep Research
            research_content = await self._conduct_deep_research(request)

            # Stage 2: Structured Output
            rating_data = await self._convert_to_structured_output(
                research_content, request
            )

            # Validate the response
            validation_result = validate_sector_rating(rating_data)
            if not validation_result.is_valid:
                raise SectorAnalysisError(
                    self.llm_model.model_name,
                    f"Invalid response format: {validation_result.errors}",
                )

            # Process references if present
            if "references" in rating_data and rating_data["references"]:
                await self._process_references(rating_data["references"], request)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            return ModelResult(
                model=self.llm_model.model_name,
                data=rating_data,  # type: ignore (validated as SectorRating)
                latency_ms=latency_ms,
                timestamp_utc=timestamp_utc,
                success=True,
            )

        except asyncio.TimeoutError:
            latency_ms = self.llm_model.timeout_seconds * 1000
            raise SectorAnalysisError(
                self.llm_model.model_name,
                f"Analysis timed out after {self.llm_model.timeout_seconds} seconds",
            )
        except LLMError as e:
            latency_ms = (time.time() - start_time) * 1000
            raise SectorAnalysisError(e.model, f"LLM error: {str(e)}", e)
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            raise SectorAnalysisError(
                self.llm_model.model_name, f"Two-stage analysis failed: {str(e)}", e
            )

    async def _conduct_deep_research(self, request: SectorRequest) -> str:
        """Stage 1: Use enhanced reasoning for comprehensive research.

        Args:
            request: Sector analysis request.

        Returns:
            Raw research content with comprehensive analysis.

        Raises:
            SectorAnalysisError: If research fails.
        """
        try:
            # Build deep research prompts with enhanced methodology
            system_prompt = build_deep_research_system_prompt()
            user_prompt = build_deep_research_user_prompt(request)

            # Use enhanced reasoning model for comprehensive analysis
            response = await self.llm_model.generate_enhanced_reasoning(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=16000,
            )

            # Extract research content
            content = response.content
            if not content:
                raise SectorAnalysisError(
                    self.llm_model.model_name, "Empty research response"
                )

            return content

        except LLMError as e:
            raise SectorAnalysisError(
                e.model, f"Deep research stage failed: {str(e)}", e
            )
        except Exception as e:
            raise SectorAnalysisError(
                self.llm_model.model_name, f"Deep research stage failed: {str(e)}", e
            )

    async def _convert_to_structured_output(
        self, research_content: str, request: SectorRequest
    ) -> Dict[str, Any]:
        """Stage 2: Convert research into structured JSON.

        Args:
            research_content: Raw research content from stage 1.
            request: Original sector request.

        Returns:
            Structured sector rating data.

        Raises:
            SectorAnalysisError: If conversion fails.
        """
        try:
            # Build structured output prompt
            system_prompt = build_structured_conversion_system_prompt()
            conversion_prompt = build_conversion_prompt(research_content, request)
            schema = create_openai_structured_output_schema()

            # Use structured output generation
            response = await self.llm_model.generate_structured(
                system_prompt=system_prompt,
                user_prompt=conversion_prompt,
                schema=schema,
                temperature=0.1,
                max_tokens=4000,
            )

            # Extract structured response
            rating_data = response.content
            if not rating_data:
                raise SectorAnalysisError(
                    self.llm_model.model_name, "Empty structured output response"
                )

            return rating_data

        except LLMError as e:
            raise SectorAnalysisError(
                e.model, f"Structured output conversion failed: {str(e)}", e
            )
        except Exception as e:
            raise SectorAnalysisError(
                self.llm_model.model_name,
                f"Structured output conversion failed: {str(e)}",
                e,
            )

    async def _process_references(
        self, references: List[Dict[str, Any]], request: SectorRequest
    ) -> None:
        """Download and archive reference URLs.

        Args:
            references: List of reference items with URLs.
            request: Original sector request for creating analysis ID.
        """
        # Create analysis ID for this request
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sector_clean = request.sector.replace(" ", "_").upper()
        analysis_id = f"{timestamp}_{sector_clean}_{request.horizon_weeks}W"

        # Create references directory
        ref_dir = Path("logs/audit/references") / analysis_id
        ref_dir.mkdir(parents=True, exist_ok=True)

        # Process each reference
        for i, ref in enumerate(references):
            if not isinstance(ref, dict) or "url" not in ref:
                continue

            url = ref["url"]
            try:
                # Validate and download URL content
                is_accessible, content = await self._download_url_content(url)

                # Update reference accessibility status
                ref["accessible"] = is_accessible
                ref["accessed_at"] = datetime.now(timezone.utc).isoformat()

                # Save content if successfully downloaded
                if is_accessible and content:
                    filename = f"ref_{i + 1:02d}_{self._sanitize_filename(ref.get('title', 'unknown'))}.html"
                    file_path = ref_dir / filename

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)

                    # Add local file path to reference
                    ref["local_path"] = str(file_path)

            except Exception as e:
                # Mark as inaccessible on error
                ref["accessible"] = False
                ref["accessed_at"] = datetime.now(timezone.utc).isoformat()
                ref["error"] = str(e)

    async def _download_url_content(self, url: str) -> tuple[bool, Optional[str]]:
        """Download content from URL with validation.

        Args:
            url: URL to download.

        Returns:
            Tuple of (is_accessible, content).
        """
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "Mozilla/5.0 (compatible; SectorAnalysis/1.0)"},
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return True, content
                    else:
                        return False, None
        except Exception:
            return False, None

    def _sanitize_filename(self, title: str) -> str:
        """Sanitize title for use as filename.

        Args:
            title: Original title string.

        Returns:
            Sanitized filename string.
        """
        # Remove or replace invalid filename characters
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", title)
        sanitized = re.sub(r"\s+", "_", sanitized)
        return sanitized[:50]  # Limit length


class SectorAdapterFactory:
    """Factory for creating sector analysis adapters.

    This factory determines the appropriate adapter type based on the
    underlying LLM model and wraps it with sector-specific functionality.
    """

    @staticmethod
    def create_adapter(
        model_name: ModelName, timeout_seconds: int = 300
    ) -> SectorAnalysisAdapter:
        """Create a sector analysis adapter for the specified model.

        Args:
            model_name: The model to create an adapter for.
            timeout_seconds: Timeout configuration.

        Returns:
            Configured SectorAnalysisAdapter instance.

        Raises:
            SectorAnalysisError: If adapter creation fails.
        """
        try:
            # Create the underlying LLM model
            llm_model = create_model(model_name, timeout_seconds)

            # Determine adapter type based on model type
            if isinstance(llm_model, TwoStageLLMModel):
                return DeepResearchSectorAdapter(llm_model)
            elif isinstance(llm_model, StandardLLMModel):
                return StandardSectorAdapter(llm_model)
            else:
                raise SectorAnalysisError(
                    model_name, f"Unsupported LLM model type: {type(llm_model)}"
                )

        except LLMError as e:
            raise SectorAnalysisError(e.model, f"Adapter creation failed: {str(e)}", e)
        except Exception as e:
            raise SectorAnalysisError(
                model_name, f"Adapter creation failed: {str(e)}", e
            )

    @staticmethod
    def create_default_adapter(timeout_seconds: int = 300) -> SectorAnalysisAdapter:
        """Create the default sector analysis adapter.

        Args:
            timeout_seconds: Timeout configuration.

        Returns:
            Default SectorAnalysisAdapter instance.

        Raises:
            SectorAnalysisError: If no adapters are available.
        """
        try:
            from ..llm_models import create_default_model

            llm_model = create_default_model(timeout_seconds)

            # Create appropriate adapter
            if isinstance(llm_model, TwoStageLLMModel):
                return DeepResearchSectorAdapter(llm_model)
            elif isinstance(llm_model, StandardLLMModel):
                return StandardSectorAdapter(llm_model)
            else:
                raise SectorAnalysisError(
                    llm_model.model_name,
                    f"Unsupported default model type: {type(llm_model)}",
                )

        except LLMError as e:
            raise SectorAnalysisError(
                e.model, f"Default adapter creation failed: {str(e)}", e
            )
        except Exception as e:
            raise SectorAnalysisError(
                ModelName.OPENAI_O4_MINI_DEEP_RESEARCH,
                f"Default adapter creation failed: {str(e)}",
                e,
            )
