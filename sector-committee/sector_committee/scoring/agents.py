"""Sector analysis agent implementation.

This module contains the main SectorAgent class that provides the primary interface
for sector analysis. The agent orchestrates the multi-model ensemble process,
handles request validation, and ensures comprehensive audit logging.

The SectorAgent serves as the main entry point for Phase 1 functionality,
providing a clean abstraction over the complex multi-model analysis pipeline.
"""

import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from ..data_models import (
    SectorRequest,
    SectorRating,
    SectorName,
    SECTOR_ETF_MAP,
    DEFAULT_WEIGHTS,
)
from .factory import ModelFactory, ResearchError

# Removed ensemble imports - using single model approach
from .schema import validate_sector_rating
from .audit import AuditLogger


class SectorAnalysisError(Exception):
    """Exception raised during sector analysis."""

    def __init__(
        self, message: str, sector: str, original_error: Optional[Exception] = None
    ):
        self.sector = sector
        self.original_error = original_error
        super().__init__(f"Sector analysis failed for {sector}: {message}")


class SectorAgent:
    """Primary interface for sector analysis using multi-model ensemble.

    This class provides the main entry point for analyzing sectors using the
    sophisticated multi-agent system. It handles request validation, orchestrates
    the ensemble analysis process, and ensures comprehensive audit logging.

    Example:
        agent = SectorAgent()
        rating = await agent.analyze_sector("Information Technology")
        print(f"XLK Rating: {rating['rating']}/5")
    """

    def __init__(
        self,
        timeout_seconds: int = 300,
        enable_audit: bool = True,
        min_confidence: float = 0.3,
    ):
        """Initialize the sector analysis agent.

        Args:
            timeout_seconds: Maximum time to wait for analysis (default: 5 minutes).
            enable_audit: Whether to enable comprehensive audit logging.
            min_confidence: Minimum confidence threshold for results.
        """
        self.timeout_seconds = timeout_seconds
        self.enable_audit = enable_audit
        self.min_confidence = min_confidence

        # Initialize research client (single model approach)
        self.research_client = ModelFactory.create_default_client(timeout_seconds)

        if self.enable_audit:
            self.audit_logger = AuditLogger()
        else:
            self.audit_logger = None

    async def analyze_sector(
        self,
        sector_name: str,
        horizon_weeks: int = 4,
        weights_hint: Optional[Dict[str, float]] = None,
    ) -> SectorRating:
        """Analyze a sector and return a comprehensive rating.

        This is the primary method for sector analysis. It validates inputs,
        orchestrates the multi-model ensemble process, validates outputs,
        and logs the complete audit trail.

        Args:
            sector_name: Name of the sector to analyze (must be valid SPDR sector).
            horizon_weeks: Investment horizon in weeks (1-52, default: 4).
            weights_hint: Optional pillar weight overrides (must sum to 1.0).

        Returns:
            SectorRating with complete analysis results.

        Raises:
            SectorAnalysisError: If analysis fails or validation errors occur.
        """
        try:
            # Validate and create request
            request = self._create_and_validate_request(
                sector_name, horizon_weeks, weights_hint
            )

            # Log analysis start
            analysis_id = self._generate_analysis_id(request)
            if self.audit_logger:
                await self.audit_logger.log_analysis_start(analysis_id, request)

            # Perform single model analysis
            model_result = await self._perform_analysis(request)

            # Validate final result
            final_rating = self._validate_and_enhance_result(model_result.data, request)

            # Apply confidence filtering
            if final_rating["confidence"] < self.min_confidence:
                raise SectorAnalysisError(
                    f"Analysis confidence {final_rating['confidence']:.2f} below "
                    f"minimum threshold {self.min_confidence:.2f}",
                    sector_name,
                )

            # Log individual model result for detailed attribution
            if self.audit_logger:
                await self.audit_logger.log_individual_model_result(
                    analysis_id, model_result, sector_name
                )

            # Log successful completion
            if self.audit_logger:
                await self.audit_logger.log_analysis_completion(
                    analysis_id, final_rating, model_result
                )

            return final_rating

        except ResearchError as e:
            # Log analysis failure
            if self.audit_logger:
                await self.audit_logger.log_analysis_failure(
                    analysis_id if "analysis_id" in locals() else "unknown",
                    sector_name,
                    str(e),
                )
            raise SectorAnalysisError(str(e), sector_name, e)

        except Exception as e:
            # Log unexpected errors
            if self.audit_logger:
                await self.audit_logger.log_analysis_failure(
                    analysis_id if "analysis_id" in locals() else "unknown",
                    sector_name,
                    f"Unexpected error: {str(e)}",
                )
            raise SectorAnalysisError(f"Unexpected error: {str(e)}", sector_name, e)

    async def analyze_multiple_sectors(
        self,
        sector_names: list[str],
        horizon_weeks: int = 4,
        weights_hint: Optional[Dict[str, float]] = None,
        max_concurrent: int = 3,
    ) -> Dict[str, SectorRating]:
        """Analyze multiple sectors concurrently.

        Args:
            sector_names: List of sector names to analyze.
            horizon_weeks: Investment horizon in weeks.
            weights_hint: Optional pillar weight overrides.
            max_concurrent: Maximum number of concurrent analyses.

        Returns:
            Dictionary mapping sector names to their ratings.

        Raises:
            SectorAnalysisError: If any analysis fails critically.
        """
        # Create semaphore to limit concurrent analyses
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_single_sector(
            sector: str,
        ) -> tuple[str, Optional[SectorRating]]:
            """Analyze single sector with concurrency control."""
            async with semaphore:
                try:
                    rating = await self.analyze_sector(
                        sector, horizon_weeks, weights_hint
                    )
                    return sector, rating
                except Exception as e:
                    # Log error but continue with other sectors
                    print(f"Failed to analyze {sector}: {e}")
                    return sector, None

        # Run analyses concurrently
        tasks = [analyze_single_sector(sector) for sector in sector_names]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Filter successful results
        successful_results = {
            sector: rating for sector, rating in results if rating is not None
        }

        return successful_results

    def _create_and_validate_request(
        self,
        sector_name: str,
        horizon_weeks: int,
        weights_hint: Optional[Dict[str, float]],
    ) -> SectorRequest:
        """Create and validate sector analysis request.

        Args:
            sector_name: Name of the sector to analyze.
            horizon_weeks: Investment horizon in weeks.
            weights_hint: Optional pillar weight overrides.

        Returns:
            Validated SectorRequest object.

        Raises:
            SectorAnalysisError: If validation fails.
        """
        # Validate sector name
        if not self._is_valid_sector(sector_name):
            valid_sectors = [s.value for s in SectorName]
            raise SectorAnalysisError(
                f"Invalid sector '{sector_name}'. Must be one of: {valid_sectors}",
                sector_name,
            )

        # Use default weights if none provided
        if weights_hint is None:
            weights_hint = DEFAULT_WEIGHTS.copy()

        try:
            # Create request (this will validate parameters)
            return SectorRequest(
                sector=sector_name,
                horizon_weeks=horizon_weeks,
                weights_hint=weights_hint,
            )
        except ValueError as e:
            raise SectorAnalysisError(
                f"Invalid request parameters: {e}", sector_name, e
            )

    async def _perform_analysis(self, request: SectorRequest) -> Any:
        """Perform single model analysis with error handling.

        Args:
            request: Validated sector analysis request.

        Returns:
            ModelResult with analysis results.

        Raises:
            SectorAnalysisError: If analysis fails.
        """
        try:
            # Perform analysis using single model approach
            return await self.research_client.analyze_sector(request)

        except ResearchError as e:
            raise SectorAnalysisError(f"Analysis failed: {str(e)}", request.sector, e)
        except Exception as e:
            raise SectorAnalysisError(
                f"Unexpected analysis error: {str(e)}", request.sector, e
            )

    def _validate_and_enhance_result(
        self, rating: SectorRating, request: SectorRequest
    ) -> SectorRating:
        """Validate and enhance the final analysis result.

        Args:
            rating: Raw sector rating from ensemble.
            request: Original analysis request.

        Returns:
            Enhanced and validated SectorRating.

        Raises:
            SectorAnalysisError: If validation fails.
        """
        # Validate against JSON schema
        validation_result = validate_sector_rating(rating)
        if not validation_result.is_valid:
            raise SectorAnalysisError(
                f"Result validation failed: {validation_result.errors}", request.sector
            )

        # Log validation warnings
        if validation_result.warnings and self.audit_logger:
            for warning in validation_result.warnings:
                print(f"Validation warning for {request.sector}: {warning}")

        # Return the validated rating without additional metadata
        # to maintain schema compliance (metadata can be logged separately)
        return rating

    def _is_valid_sector(self, sector_name: str) -> bool:
        """Check if sector name is valid SPDR sector.

        Args:
            sector_name: Sector name to validate.

        Returns:
            True if sector is valid, False otherwise.
        """
        valid_sectors = {s.value for s in SectorName}
        return sector_name in valid_sectors

    def _get_sector_etf(self, sector_name: str) -> str:
        """Get ETF ticker for sector.

        Args:
            sector_name: Sector name.

        Returns:
            ETF ticker symbol.
        """
        for sector_enum in SectorName:
            if sector_enum.value == sector_name:
                return SECTOR_ETF_MAP[sector_enum]

        return "UNKNOWN"  # Fallback (should not happen with validation)

    def _generate_analysis_id(self, request: SectorRequest) -> str:
        """Generate unique analysis ID for audit logging.

        Args:
            request: Sector analysis request.

        Returns:
            Unique analysis identifier.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sector_code = request.sector.replace(" ", "_").upper()
        return f"{timestamp}_{sector_code}_{request.horizon_weeks}W"

    async def get_supported_sectors(self) -> list[str]:
        """Get list of supported sector names.

        Returns:
            List of valid sector names.
        """
        return [sector.value for sector in SectorName]

    async def get_sector_etf_mapping(self) -> Dict[str, str]:
        """Get mapping of sectors to ETF tickers.

        Returns:
            Dictionary mapping sector names to ETF tickers.
        """
        return {sector.value: etf for sector, etf in SECTOR_ETF_MAP.items()}

    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check.

        Returns:
            Health status information.
        """
        health_status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": self.timeout_seconds,
            "audit_enabled": self.enable_audit,
            "min_confidence": self.min_confidence,
            "supported_sectors": len(SectorName),
            "supported_models": len(ModelFactory.get_supported_models()),
        }

        # Test model availability
        try:
            test_client = ModelFactory.create_default_client(timeout_seconds=10)
            model_name = test_client.get_model_name().value
            health_status["models_available"] = 1
            health_status["available_models"] = [model_name]
            health_status["status"] = "healthy"
        except Exception as e:
            health_status["models_available"] = 0
            health_status["available_models"] = []
            health_status["status"] = "degraded"
            health_status["error"] = str(e)

        return health_status


# Convenience functions for common use cases
async def analyze_sector_quick(
    sector_name: str, horizon_weeks: int = 4
) -> SectorRating:
    """Quick sector analysis with default settings.

    Args:
        sector_name: Name of the sector to analyze.
        horizon_weeks: Investment horizon in weeks.

    Returns:
        SectorRating with analysis results.
    """
    agent = SectorAgent()
    return await agent.analyze_sector(sector_name, horizon_weeks)


async def analyze_all_sectors(
    horizon_weeks: int = 4, max_concurrent: int = 3
) -> Dict[str, SectorRating]:
    """Analyze all supported sectors.

    Args:
        horizon_weeks: Investment horizon in weeks.
        max_concurrent: Maximum concurrent analyses.

    Returns:
        Dictionary mapping sector names to ratings.
    """
    agent = SectorAgent()
    all_sectors = await agent.get_supported_sectors()
    return await agent.analyze_multiple_sectors(
        all_sectors, horizon_weeks, max_concurrent=max_concurrent
    )
