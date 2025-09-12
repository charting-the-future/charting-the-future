"""Ensemble aggregation for multi-model sector analysis.

This module implements ensemble methods for combining predictions from multiple
LLM models to produce consensus sector ratings. The ensemble approach improves
reliability and reduces single-model bias through sophisticated aggregation
techniques.

The aggregation process includes weighted voting, confidence-based weighting,
and consistency validation to ensure high-quality final ratings.
"""

import asyncio
import statistics
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from ..data_models import (
    ModelResult,
    EnsembleResult,
    SectorRating,
    SectorRequest,
    SCORE_RANGE,
    CONFIDENCE_RANGE,
)
from .factory import ResearchClient


class EnsembleError(Exception):
    """Exception raised during ensemble aggregation."""

    def __init__(self, message: str, model_errors: Optional[List[str]] = None):
        self.model_errors = model_errors or []
        super().__init__(message)


class EnsembleAggregator:
    """Aggregates results from multiple research models.

    This class implements sophisticated ensemble methods for combining sector
    analysis results from multiple LLM models, providing more reliable and
    robust predictions than any single model alone.
    """

    def __init__(
        self, min_models: int = 1, consensus_threshold: float = 0.8, audit_logger=None
    ):
        """Initialize the ensemble aggregator.

        Args:
            min_models: Minimum number of successful models required.
            consensus_threshold: Minimum agreement level (0-1) for high confidence.
            audit_logger: Optional audit logger for individual model tracking.
        """
        self.min_models = min_models
        self.consensus_threshold = consensus_threshold
        self.audit_logger = audit_logger

    async def analyze_sector_ensemble(
        self, clients: List[ResearchClient], request: SectorRequest
    ) -> EnsembleResult:
        """Analyze sector using ensemble of research clients.

        Args:
            clients: List of research clients to use.
            request: Sector analysis request.

        Returns:
            EnsembleResult with aggregated analysis.

        Raises:
            EnsembleError: If insufficient models succeed or aggregation fails.
        """
        if len(clients) < self.min_models:
            raise EnsembleError(
                f"Insufficient clients: need at least {self.min_models}, got {len(clients)}"
            )

        # Run all models in parallel
        model_results = await self._run_models_parallel(clients, request)

        # Log individual model results for detailed attribution (if audit logger available)
        if self.audit_logger:
            # Generate analysis ID for this ensemble run
            analysis_id = self._generate_ensemble_analysis_id(request)

            # Log each individual model result
            for model_result in model_results:
                await self.audit_logger.log_individual_model_result(
                    analysis_id, model_result, request.sector
                )

        # Filter successful results
        successful_results = [r for r in model_results if r.success]

        if len(successful_results) < self.min_models:
            errors = [r.error_message for r in model_results if not r.success]
            raise EnsembleError(
                f"Insufficient successful models: need {self.min_models}, got {len(successful_results)}",
                errors,
            )

        # Aggregate the results
        final_rating = self._aggregate_ratings(successful_results)

        # Calculate consensus metrics
        consensus_score = self._calculate_consensus(successful_results)

        # Calculate total metrics
        total_latency = max(r.latency_ms for r in model_results)
        total_cost = sum(r.cost_usd for r in model_results)
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        return EnsembleResult(
            final_rating=final_rating,
            model_results=model_results,
            consensus_score=consensus_score,
            total_latency_ms=total_latency,
            total_cost_usd=total_cost,
            timestamp_utc=timestamp_utc,
        )

    async def _run_models_parallel(
        self, clients: List[ResearchClient], request: SectorRequest
    ) -> List[ModelResult]:
        """Run multiple models in parallel with error handling.

        Args:
            clients: List of research clients.
            request: Sector analysis request.

        Returns:
            List of ModelResult objects (including failed ones).
        """

        async def run_single_model(client: ResearchClient) -> ModelResult:
            """Run single model with error handling."""
            try:
                return await client.analyze_sector(request)
            except Exception as e:
                # Create failed result
                return ModelResult(
                    model=client.get_model_name(),
                    data={},  # type: ignore (will not be used for failed results)
                    latency_ms=0.0,
                    cost_usd=0.0,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    success=False,
                    error_message=str(e),
                )

        # Run all models concurrently
        tasks = [run_single_model(client) for client in clients]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        return results

    def _aggregate_ratings(self, results: List[ModelResult]) -> SectorRating:
        """Aggregate multiple model results into final rating.

        Args:
            results: List of successful model results.

        Returns:
            Aggregated SectorRating.
        """
        if not results:
            raise EnsembleError("No results to aggregate")

        # Extract data for aggregation
        ratings = [r.data["rating"] for r in results]
        sub_scores = [r.data["sub_scores"] for r in results]
        confidences = [r.data["confidence"] for r in results]
        weights = [r.data["weights"] for r in results]

        # Aggregate ratings using confidence-weighted voting
        final_rating = self._aggregate_ratings_weighted(ratings, confidences)

        # Aggregate sub-scores
        final_sub_scores = {
            "fundamentals": self._aggregate_scores(
                [s["fundamentals"] for s in sub_scores], confidences
            ),
            "sentiment": self._aggregate_scores(
                [s["sentiment"] for s in sub_scores], confidences
            ),
            "technicals": self._aggregate_scores(
                [s["technicals"] for s in sub_scores], confidences
            ),
        }

        # Average weights (simple average since they should be similar)
        final_weights = {
            "fundamentals": statistics.mean([w["fundamentals"] for w in weights]),
            "sentiment": statistics.mean([w["sentiment"] for w in weights]),
            "technicals": statistics.mean([w["technicals"] for w in weights]),
        }

        # Calculate weighted score
        final_weighted_score = (
            final_sub_scores["fundamentals"] * final_weights["fundamentals"]
            + final_sub_scores["sentiment"] * final_weights["sentiment"]
            + final_sub_scores["technicals"] * final_weights["technicals"]
        )

        # Aggregate rationale and references
        final_rationale = self._aggregate_rationale(results)
        final_references = self._aggregate_references(results)

        # Calculate ensemble confidence
        final_confidence = self._calculate_ensemble_confidence(results)

        # Generate summary
        final_summary = self._generate_ensemble_summary(
            final_rating, final_confidence, results
        )

        return {
            "rating": final_rating,
            "summary": final_summary,
            "sub_scores": final_sub_scores,  # type: ignore
            "weights": final_weights,  # type: ignore
            "weighted_score": final_weighted_score,
            "rationale": final_rationale,
            "references": final_references,
            "confidence": final_confidence,
        }

    def _aggregate_ratings_weighted(
        self, ratings: List[int], confidences: List[float]
    ) -> int:
        """Aggregate ratings using confidence weighting.

        Args:
            ratings: List of integer ratings (1-5).
            confidences: List of confidence scores (0-1).

        Returns:
            Aggregated integer rating.
        """
        if len(ratings) != len(confidences):
            raise EnsembleError("Ratings and confidences must have same length")

        # Calculate confidence-weighted average
        weighted_sum = sum(r * c for r, c in zip(ratings, confidences))
        weight_sum = sum(confidences)

        if weight_sum == 0:
            # Fallback to simple average if all confidences are 0
            weighted_avg = statistics.mean(ratings)
        else:
            weighted_avg = weighted_sum / weight_sum

        # Round to nearest integer and clamp to valid range
        final_rating = max(SCORE_RANGE[0], min(SCORE_RANGE[1], round(weighted_avg)))
        return final_rating

    def _aggregate_scores(self, scores: List[int], confidences: List[float]) -> int:
        """Aggregate sub-scores using confidence weighting.

        Args:
            scores: List of sub-scores (1-5).
            confidences: List of confidence values.

        Returns:
            Aggregated sub-score.
        """
        return self._aggregate_ratings_weighted(scores, confidences)

    def _aggregate_rationale(self, results: List[ModelResult]) -> List[Dict[str, Any]]:
        """Aggregate rationale items from multiple models.

        Args:
            results: List of model results.

        Returns:
            Combined rationale list.
        """
        all_rationale = []

        for result in results:
            for item in result.data["rationale"]:
                # Keep original rationale without model attribution
                # to maintain schema compliance
                all_rationale.append(item.copy())

        # Sort by confidence and limit to top items
        all_rationale.sort(key=lambda x: x["confidence"], reverse=True)
        return all_rationale[:10]  # Limit to top 10 items

    def _aggregate_references(self, results: List[ModelResult]) -> List[Dict[str, Any]]:
        """Aggregate reference items from multiple models.

        Args:
            results: List of model results.

        Returns:
            Combined and deduplicated reference list.
        """
        seen_urls = set()
        all_references = []

        for result in results:
            for ref in result.data["references"]:
                url = ref["url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_references.append(ref)

        # Sort by accessibility and recency
        all_references.sort(
            key=lambda x: (x["accessible"], x["accessed_at"]), reverse=True
        )

        return all_references[:10]  # Limit to top 10 references

    def _calculate_ensemble_confidence(self, results: List[ModelResult]) -> float:
        """Calculate ensemble confidence based on model agreement.

        Args:
            results: List of model results.

        Returns:
            Ensemble confidence score (0-1).
        """
        if len(results) == 1:
            return results[0].data["confidence"]

        # Calculate agreement between models
        ratings = [r.data["rating"] for r in results]
        rating_std = statistics.stdev(ratings) if len(ratings) > 1 else 0.0

        # Calculate average confidence
        avg_confidence = statistics.mean([r.data["confidence"] for r in results])

        # Adjust confidence based on agreement
        # Lower standard deviation = higher agreement = higher ensemble confidence
        max_std = 2.0  # Maximum possible standard deviation for ratings 1-5
        agreement_factor = 1.0 - (rating_std / max_std)

        # Combine average confidence with agreement factor
        ensemble_confidence = avg_confidence * (0.7 + 0.3 * agreement_factor)

        # Clamp to valid range
        return max(CONFIDENCE_RANGE[0], min(CONFIDENCE_RANGE[1], ensemble_confidence))

    def _calculate_consensus(self, results: List[ModelResult]) -> float:
        """Calculate consensus score between models.

        Args:
            results: List of model results.

        Returns:
            Consensus score (0-1), where 1.0 means perfect agreement.
        """
        if len(results) <= 1:
            return 1.0

        ratings = [r.data["rating"] for r in results]

        # Calculate pairwise agreement
        agreements = []
        for i in range(len(ratings)):
            for j in range(i + 1, len(ratings)):
                # Agreement is inversely related to absolute difference
                diff = abs(ratings[i] - ratings[j])
                max_diff = (
                    SCORE_RANGE[1] - SCORE_RANGE[0]
                )  # Maximum possible difference
                agreement = 1.0 - (diff / max_diff)
                agreements.append(agreement)

        return statistics.mean(agreements) if agreements else 1.0

    def _generate_ensemble_summary(
        self, rating: int, confidence: float, results: List[ModelResult]
    ) -> str:
        """Generate a summary for the ensemble result.

        Args:
            rating: Final ensemble rating.
            confidence: Final ensemble confidence.
            results: List of model results.

        Returns:
            Summary string.
        """
        model_count = len(results)
        model_names = [
            r.model.value.split("-")[0] for r in results
        ]  # Extract base model names

        rating_labels = {
            1: "Very Bearish",
            2: "Bearish",
            3: "Neutral",
            4: "Bullish",
            5: "Very Bullish",
        }
        rating_label = rating_labels.get(rating, "Unknown")

        confidence_pct = int(confidence * 100)

        return (
            f"Ensemble {rating_label} ({rating}/5) rating with {confidence_pct}% confidence "
            f"based on consensus from {model_count} models ({', '.join(set(model_names))}). "
            f"Aggregated using confidence-weighted voting across tri-pillar analysis."
        )

    def _generate_ensemble_analysis_id(self, request: SectorRequest) -> str:
        """Generate unique analysis ID for ensemble audit logging.

        Args:
            request: Sector analysis request.

        Returns:
            Unique ensemble analysis identifier.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sector_code = request.sector.replace(" ", "_").upper()
        return f"{timestamp}_{sector_code}_{request.horizon_weeks}W_ENSEMBLE"


def create_default_ensemble() -> EnsembleAggregator:
    """Create ensemble aggregator with default settings.

    Returns:
        Configured EnsembleAggregator instance.
    """
    return EnsembleAggregator(
        min_models=1,  # Allow single model for development
        consensus_threshold=0.8,  # Require 80% agreement for high confidence
    )


async def quick_ensemble_analysis(
    request: SectorRequest, timeout_seconds: int = 300
) -> EnsembleResult:
    """Perform quick ensemble analysis with default settings.

    Args:
        request: Sector analysis request.
        timeout_seconds: Timeout for each model.

    Returns:
        EnsembleResult with aggregated analysis.
    """
    from .factory import ModelFactory

    # Create ensemble clients
    clients = ModelFactory.create_ensemble_clients(timeout_seconds)

    # Create aggregator
    aggregator = create_default_ensemble()

    # Run ensemble analysis
    return await aggregator.analyze_sector_ensemble(clients, request)
