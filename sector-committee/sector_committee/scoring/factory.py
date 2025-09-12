"""Model factory for sector analysis providers.

This module provides a thin wrapper around the new llm_adapters module,
maintaining backward compatibility while delegating to the centralized
sector analysis adapter system.

The factory now uses the llm_adapters module for all model creation and
management, providing clean separation of concerns with proper abstraction layers.
"""

import abc
from typing import List, Any, Optional

from ..data_models import SectorRequest, ModelResult
from ..llm_models import ModelName
from ..llm_models import get_supported_models
from .llm_adapters import (
    SectorAnalysisAdapter,
    SectorAnalysisError,
    SectorAdapterFactory,
)


class ResearchClient(abc.ABC):
    """Abstract interface for sector research clients.

    This abstract base class defines the interface that all research client
    implementations must follow, ensuring consistent behavior regardless of
    the underlying LLM provider.

    Note: This interface is maintained for backward compatibility.
    New code should use llm_models.LLMModel directly.
    """

    @abc.abstractmethod
    async def analyze_sector(self, request: SectorRequest) -> ModelResult:
        """Analyze a sector and return a structured rating.

        Args:
            request: Sector analysis request with parameters.

        Returns:
            ModelResult containing the analysis and metadata.

        Raises:
            ResearchError: If analysis fails or times out.
        """
        pass

    @abc.abstractmethod
    def get_model_name(self) -> ModelName:
        """Get the model name for this client.

        Returns:
            ModelName enum value for this client.
        """
        pass


class ResearchError(Exception):
    """Exception raised when sector research fails.

    Attributes:
        model: The model that failed.
        message: Description of the error.
        original_error: The underlying exception (if any).

    Note: This is maintained for backward compatibility.
    New code should use llm_adapters.SectorAnalysisError directly.
    """

    def __init__(
        self, model: ModelName, message: str, original_error: Optional[Exception] = None
    ):
        self.model = model
        self.original_error = original_error
        super().__init__(f"{model.value}: {message}")


class SectorAdapterWrapper(ResearchClient):
    """Adapter to wrap SectorAnalysisAdapter as ResearchClient.

    This adapter provides backward compatibility by wrapping the new
    SectorAnalysisAdapter interface to match the old ResearchClient interface.
    """

    def __init__(self, sector_adapter: SectorAnalysisAdapter):
        """Initialize the adapter wrapper.

        Args:
            sector_adapter: The SectorAnalysisAdapter instance to wrap.
        """
        self.sector_adapter = sector_adapter

    async def analyze_sector(self, request: SectorRequest) -> ModelResult:
        """Analyze a sector using the wrapped SectorAnalysisAdapter.

        Args:
            request: Sector analysis request.

        Returns:
            ModelResult from the SectorAnalysisAdapter.

        Raises:
            ResearchError: If analysis fails.
        """
        try:
            return await self.sector_adapter.analyze_sector(request)
        except SectorAnalysisError as e:
            raise ResearchError(e.model, str(e), e.original_error)

    def get_model_name(self) -> ModelName:
        """Get the model name from the wrapped SectorAnalysisAdapter.

        Returns:
            ModelName enum value.
        """
        return self.sector_adapter.get_model_name()


class ModelFactory:
    """Factory for creating research clients.

    This factory provides a clean interface while delegating to the
    llm_adapters module for all model management.

    Note: This class is maintained for backward compatibility.
    New code should use llm_adapters.SectorAdapterFactory directly.
    """

    @staticmethod
    def create(model: ModelName, **kwargs: Any) -> ResearchClient:
        """Create a research client for the specified model.

        Args:
            model: The model to create a client for.
            **kwargs: Additional configuration parameters.

        Returns:
            Configured ResearchClient instance.

        Raises:
            ResearchError: If client creation fails.
        """
        try:
            # Extract timeout_seconds from kwargs, default to 300
            timeout_seconds = kwargs.get("timeout_seconds", 300)

            # Use the new sector adapter system
            sector_adapter = SectorAdapterFactory.create_adapter(model, timeout_seconds)
            return SectorAdapterWrapper(sector_adapter)
        except SectorAnalysisError as e:
            raise ResearchError(e.model, str(e), e.original_error)

    @staticmethod
    def get_supported_models() -> List[ModelName]:
        """Get list of supported models.

        Returns:
            List of supported ModelName values.
        """
        return get_supported_models()

    @staticmethod
    def create_default_client(timeout_seconds: int = 300) -> ResearchClient:
        """Create the default research client.

        Args:
            timeout_seconds: Timeout for the client.

        Returns:
            Default research client instance.
        """
        try:
            sector_adapter = SectorAdapterFactory.create_default_adapter(
                timeout_seconds
            )
            return SectorAdapterWrapper(sector_adapter)
        except SectorAnalysisError as e:
            raise ResearchError(e.model, str(e), e.original_error)
