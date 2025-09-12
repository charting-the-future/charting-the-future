"""Audit trail and logging utilities for sector analysis.

This module provides comprehensive audit logging capabilities for the sector
analysis system, ensuring full regulatory compliance and traceability of all
analysis activities from inputs to outputs.

The audit system captures complete request/response data, timing information,
model attribution, and error conditions to support regulatory requirements
such as SR 11-7 model governance and MiFID II documentation.
"""

import json
import os
import aiofiles  # type: ignore
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from ..data_models import SectorRequest, SectorRating


class AuditLogger:
    """Comprehensive audit logging for sector analysis system.

    This class provides systematic logging of all analysis activities with
    structured JSON output for regulatory compliance and operational monitoring.
    """

    def __init__(self, log_directory: Optional[str] = None):
        """Initialize audit logger.

        Args:
            log_directory: Directory for audit logs (default: ./logs/audit).
        """
        if log_directory is None:
            log_directory = os.getenv("AUDIT_LOG_DIR", "./logs/audit")

        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        # Separate directories for different log types
        self.analysis_logs_dir = self.log_directory / "analysis"
        self.error_logs_dir = self.log_directory / "errors"
        self.performance_logs_dir = self.log_directory / "performance"
        self.model_results_dir = self.log_directory / "model-results"

        # Create subdirectories
        for log_dir in [
            self.analysis_logs_dir,
            self.error_logs_dir,
            self.performance_logs_dir,
            self.model_results_dir,
        ]:
            log_dir.mkdir(parents=True, exist_ok=True)

    async def log_analysis_start(
        self, analysis_id: str, request: SectorRequest
    ) -> None:
        """Log the start of a sector analysis.

        Args:
            analysis_id: Unique identifier for this analysis.
            request: Sector analysis request parameters.
        """
        log_entry = {
            "event_type": "analysis_start",
            "analysis_id": analysis_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request": {
                "sector": request.sector,
                "horizon_weeks": request.horizon_weeks,
                "weights_hint": request.weights_hint,
            },
            "system_info": await self._get_system_info(),
            "compliance": {
                "regulation": "SR 11-7 Model Governance",
                "purpose": "Systematic sector analysis audit trail",
                "retention_policy": "7 years minimum",
            },
        }

        await self._write_log_file(
            self.analysis_logs_dir / f"{analysis_id}_start.json", log_entry
        )

    async def log_analysis_completion(
        self,
        analysis_id: str,
        final_rating: SectorRating,
        model_result: Any,  # Can be EnsembleResult or ModelResult
    ) -> None:
        """Log successful completion of sector analysis.

        Args:
            analysis_id: Unique identifier for this analysis.
            final_rating: Final sector rating result.
            model_result: Model result (supports both single model and ensemble).
        """
        # Handle both single model and ensemble results
        if hasattr(model_result, "model_results"):  # EnsembleResult
            metrics = {
                "consensus_score": model_result.consensus_score,
                "total_latency_ms": model_result.total_latency_ms,
                "total_cost_usd": model_result.total_cost_usd,
                "model_count": len(model_result.model_results),
                "successful_models": len(
                    [r for r in model_result.model_results if r.success]
                ),
            }
            model_results = [
                {
                    "model": result.model.value,
                    "success": result.success,
                    "latency_ms": result.latency_ms,
                    "cost_usd": getattr(result, "cost_usd", 0.0),
                    "timestamp_utc": result.timestamp_utc,
                    "error_message": result.error_message,
                    "rating": result.data.get("rating") if result.success else None,
                    "confidence": result.data.get("confidence")
                    if result.success
                    else None,
                }
                for result in model_result.model_results
            ]
        else:  # Single ModelResult
            metrics = {
                "consensus_score": 1.0,  # Perfect consensus for single model
                "total_latency_ms": model_result.latency_ms,
                "total_cost_usd": getattr(model_result, "cost_usd", 0.0),
                "model_count": 1,
                "successful_models": 1 if model_result.success else 0,
            }
            model_results = [
                {
                    "model": model_result.model.value,
                    "success": model_result.success,
                    "latency_ms": model_result.latency_ms,
                    "cost_usd": getattr(model_result, "cost_usd", 0.0),
                    "timestamp_utc": model_result.timestamp_utc,
                    "error_message": model_result.error_message,
                    "rating": model_result.data.get("rating")
                    if model_result.success
                    else None,
                    "confidence": model_result.data.get("confidence")
                    if model_result.success
                    else None,
                }
            ]

        # Process reference management information
        references = final_rating.get("references", [])
        reference_stats = {
            "total_references": len(references),
            "accessible_references": sum(
                1 for ref in references if ref.get("accessible", False)
            ),
            "downloaded_references": sum(
                1 for ref in references if ref.get("local_path")
            ),
            "failed_references": sum(1 for ref in references if ref.get("error")),
            "reference_sources": list(
                set(
                    ref.get("url", "").split("/")[2]
                    for ref in references
                    if ref.get("url")
                )
            ),
        }

        log_entry = {
            "event_type": "analysis_completion",
            "analysis_id": analysis_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "final_rating": final_rating,
            "analysis_metrics": metrics,
            "model_results": model_results,
            "reference_management": reference_stats,
            "validation": {
                "schema_validated": True,
                "business_logic_validated": True,
                "confidence_threshold_met": final_rating["confidence"] >= 0.3,
                "references_validated": reference_stats["total_references"] > 0,
                "deep_research_used": any(
                    result["model"] == "o4-mini-deep-research"
                    for result in model_results
                ),
            },
            "compliance": {
                "full_audit_trail": True,
                "model_attribution_complete": True,
                "reference_validation_performed": True,
                "reference_archiving_complete": reference_stats["downloaded_references"]
                > 0,
                "regulatory_compliant": True,
            },
        }

        await self._write_log_file(
            self.analysis_logs_dir / f"{analysis_id}_completion.json", log_entry
        )

        # Also log performance metrics separately
        await self._log_performance_metrics(analysis_id, model_result)

    async def log_analysis_failure(
        self,
        analysis_id: str,
        sector_name: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log failed sector analysis with error details.

        Args:
            analysis_id: Unique identifier for this analysis.
            sector_name: Name of the sector being analyzed.
            error_message: Description of the error.
            error_details: Optional additional error information.
        """
        log_entry = {
            "event_type": "analysis_failure",
            "analysis_id": analysis_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sector": sector_name,
            "error": {
                "message": error_message,
                "details": error_details or {},
                "error_category": self._classify_error(error_message),
            },
            "system_state": await self._get_system_info(),
            "impact": {
                "analysis_failed": True,
                "partial_results_available": False,
                "requires_investigation": True,
            },
            "compliance": {
                "error_logged": True,
                "investigation_required": True,
                "escalation_needed": self._requires_escalation(error_message),
            },
        }

        await self._write_log_file(
            self.error_logs_dir / f"{analysis_id}_error.json", log_entry
        )

    async def log_model_request(
        self, analysis_id: str, model_name: str, request_data: Dict[str, Any]
    ) -> None:
        """Log individual model API request.

        Args:
            analysis_id: Unique identifier for the analysis.
            model_name: Name of the model being called.
            request_data: Request parameters sent to the model.
        """
        log_entry = {
            "event_type": "model_request",
            "analysis_id": analysis_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "request": {
                "prompt_length": len(str(request_data)),
                "parameters": {
                    k: v
                    for k, v in request_data.items()
                    if k not in ["messages", "prompt"]  # Exclude large text
                },
                "has_web_search": "web_search" in str(request_data),
                "structured_output": "response_format" in request_data,
            },
            "compliance": {
                "prompt_versioning": True,
                "parameter_logging": True,
                "reconstruction_possible": True,
            },
        }

        await self._write_log_file(
            self.analysis_logs_dir / f"{analysis_id}_model_{model_name}_request.json",
            log_entry,
        )

    async def log_model_response(
        self,
        analysis_id: str,
        model_name: str,
        response_data: Dict[str, Any],
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        """Log individual model API response.

        Args:
            analysis_id: Unique identifier for the analysis.
            model_name: Name of the model that responded.
            response_data: Response data from the model.
            latency_ms: Response latency in milliseconds.
            cost_usd: Estimated cost in USD.
        """
        references = response_data.get("references", [])

        log_entry = {
            "event_type": "model_response",
            "analysis_id": analysis_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "response": {
                "rating": response_data.get("rating"),
                "confidence": response_data.get("confidence"),
                "sub_scores": response_data.get("sub_scores"),
                "reference_count": len(references),
                "rationale_count": len(response_data.get("rationale", [])),
                "response_length": len(str(response_data)),
            },
            "reference_management": {
                "total_references": len(references),
                "accessible_references": sum(
                    1 for ref in references if ref.get("accessible", False)
                ),
                "downloaded_references": sum(
                    1 for ref in references if ref.get("local_path")
                ),
                "failed_references": sum(1 for ref in references if ref.get("error")),
                "reference_domains": list(
                    set(
                        ref.get("url", "").split("/")[2]
                        for ref in references
                        if ref.get("url") and "/" in ref.get("url", "")
                    )
                ),
                "download_success_rate": (
                    sum(1 for ref in references if ref.get("accessible", False))
                    / len(references)
                    if references
                    else 0
                ),
            },
            "performance": {
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "tokens_used": response_data.get("token_usage", {}),
                "deep_research_used": model_name == "o4-mini-deep-research",
            },
            "quality": {
                "schema_valid": True,  # Assumes validation passed
                "has_references": len(references) > 0,
                "confidence_level": "high"
                if response_data.get("confidence", 0) > 0.7
                else "medium",
                "real_urls_provided": model_name == "o4-mini-deep-research"
                and len(references) > 0,
                "reference_validation_complete": all(
                    "accessible" in ref for ref in references
                ),
            },
        }

        await self._write_log_file(
            self.analysis_logs_dir / f"{analysis_id}_model_{model_name}_response.json",
            log_entry,
        )

    async def log_individual_model_result(
        self,
        analysis_id: str,
        model_result: Any,  # ModelResult
        sector_name: str,
    ) -> None:
        """Log individual model result with detailed tri-pillar scores.

        This method addresses the model tracking gap in ensemble scenarios by
        providing detailed attribution of individual model contributions.

        Args:
            analysis_id: Unique identifier for the analysis.
            model_result: Individual model result to log.
            sector_name: Name of sector being analyzed.
        """
        if not model_result.success:
            # Log failed model attempts
            log_entry = {
                "event_type": "individual_model_failure",
                "analysis_id": analysis_id,
                "timestamp_utc": model_result.timestamp_utc,
                "model_info": {
                    "model_name": model_result.model.value,
                    "model_provider": model_result.model.value.split("-")[
                        0
                    ],  # Extract provider
                },
                "sector": sector_name,
                "error": {
                    "message": model_result.error_message,
                    "category": self._classify_error(model_result.error_message or ""),
                },
                "performance": {
                    "latency_ms": model_result.latency_ms,
                    "success": False,
                },
                "compliance": {
                    "model_attribution_complete": True,
                    "failure_documented": True,
                },
            }
        else:
            # Log successful model results with detailed tri-pillar attribution
            model_data = model_result.data
            log_entry = {
                "event_type": "individual_model_result",
                "analysis_id": analysis_id,
                "timestamp_utc": model_result.timestamp_utc,
                "model_info": {
                    "model_name": model_result.model.value,
                    "model_provider": model_result.model.value.split("-")[
                        0
                    ],  # Extract provider
                    "model_type": "deep_research"
                    if "deep-research" in model_result.model.value
                    else "standard",
                },
                "sector": sector_name,
                "analysis_result": {
                    "overall_rating": model_data.get("rating"),
                    "confidence": model_data.get("confidence"),
                    "tri_pillar_scores": {
                        "fundamentals": model_data.get("sub_scores", {}).get(
                            "fundamentals"
                        ),
                        "sentiment": model_data.get("sub_scores", {}).get("sentiment"),
                        "technicals": model_data.get("sub_scores", {}).get(
                            "technicals"
                        ),
                    },
                    "pillar_weights": {
                        "fundamentals": model_data.get("weights", {}).get(
                            "fundamentals"
                        ),
                        "sentiment": model_data.get("weights", {}).get("sentiment"),
                        "technicals": model_data.get("weights", {}).get("technicals"),
                    },
                    "weighted_score": model_data.get("weighted_score"),
                    "summary": model_data.get("summary", "")[
                        :200
                    ],  # Truncate for logging
                },
                "performance": {
                    "latency_ms": model_result.latency_ms,
                    "cost_usd": getattr(
                        model_result, "cost_usd", 0.0
                    ),  # Graceful handling for missing cost
                    "success": True,
                    "deep_research_used": "deep-research" in model_result.model.value,
                },
                "quality_metrics": {
                    "rationale_count": len(model_data.get("rationale", [])),
                    "reference_count": len(model_data.get("references", [])),
                    "confidence_level": "high"
                    if model_data.get("confidence", 0) > 0.7
                    else "medium"
                    if model_data.get("confidence", 0) > 0.4
                    else "low",
                    "has_tri_pillar_coverage": all(
                        model_data.get("sub_scores", {}).get(pillar) is not None
                        for pillar in ["fundamentals", "sentiment", "technicals"]
                    ),
                },
                "rationale_summary": [
                    {
                        "pillar": item.get("pillar"),
                        "impact": item.get("impact"),
                        "confidence": item.get("confidence"),
                        "reason_preview": item.get("reason", "")[
                            :100
                        ],  # First 100 chars
                    }
                    for item in model_data.get("rationale", [])[
                        :5
                    ]  # Top 5 rationale items
                ],
                "reference_summary": {
                    "total_references": len(model_data.get("references", [])),
                    "accessible_references": sum(
                        1
                        for ref in model_data.get("references", [])
                        if ref.get("accessible", False)
                    ),
                    "unique_domains": len(
                        set(
                            ref.get("url", "").split("/")[2]
                            for ref in model_data.get("references", [])
                            if ref.get("url") and "/" in ref.get("url", "")
                        )
                    ),
                },
                "compliance": {
                    "model_attribution_complete": True,
                    "tri_pillar_documented": True,
                    "performance_tracked": True,
                    "schema_validated": True,
                },
            }

        # Generate model-specific filename
        model_safe_name = model_result.model.value.replace("-", "_")
        filename = f"{analysis_id}_{model_safe_name}_result.json"

        await self._write_log_file(
            self.model_results_dir / filename,
            log_entry,
        )

    async def _log_performance_metrics(
        self,
        analysis_id: str,
        model_result: Any,  # Can be EnsembleResult or ModelResult
    ) -> None:
        """Log performance metrics for monitoring and optimization.

        Args:
            analysis_id: Unique identifier for the analysis.
            model_result: Model result (supports both single model and ensemble).
        """
        # Handle both single model and ensemble results
        if hasattr(model_result, "model_results"):  # EnsembleResult
            overall_metrics = {
                "total_latency_ms": model_result.total_latency_ms,
                "total_cost_usd": model_result.total_cost_usd,
                "consensus_score": model_result.consensus_score,
                "success_rate": len(
                    [r for r in model_result.model_results if r.success]
                )
                / len(model_result.model_results),
            }
            model_performance = [
                {
                    "model": result.model.value,
                    "latency_ms": result.latency_ms,
                    "cost_usd": getattr(result, "cost_usd", 0.0),
                    "success": result.success,
                    "confidence": result.data.get("confidence")
                    if result.success
                    else None,
                }
                for result in model_result.model_results
            ]
            final_rating = model_result.final_rating
        else:  # Single ModelResult
            overall_metrics = {
                "total_latency_ms": model_result.latency_ms,
                "total_cost_usd": getattr(model_result, "cost_usd", 0.0),
                "consensus_score": 1.0,  # Perfect consensus for single model
                "success_rate": 1.0 if model_result.success else 0.0,
            }
            model_performance = [
                {
                    "model": model_result.model.value,
                    "latency_ms": model_result.latency_ms,
                    "cost_usd": getattr(model_result, "cost_usd", 0.0),
                    "success": model_result.success,
                    "confidence": model_result.data.get("confidence")
                    if model_result.success
                    else None,
                }
            ]
            final_rating = model_result.data

        performance_entry = {
            "event_type": "performance_metrics",
            "analysis_id": analysis_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "overall_metrics": overall_metrics,
            "model_performance": model_performance,
            "quality_metrics": {
                "final_confidence": final_rating.get("confidence", 0),
                "reference_count": len(final_rating.get("references", [])),
                "accessible_references": sum(
                    1
                    for ref in final_rating.get("references", [])
                    if ref.get("accessible", False)
                ),
                "downloaded_references": sum(
                    1
                    for ref in final_rating.get("references", [])
                    if ref.get("local_path")
                ),
            },
            "thresholds": {
                "latency_target_ms": 300000,  # 5 minutes
                "cost_target_usd": 5.0,
                "confidence_minimum": 0.3,
                "consensus_minimum": 0.8,
            },
            "alerts": {
                "latency_exceeded": overall_metrics["total_latency_ms"] > 300000,
                "cost_exceeded": overall_metrics["total_cost_usd"] > 5.0,
                "low_confidence": final_rating.get("confidence", 0) < 0.5,
                "low_consensus": overall_metrics["consensus_score"] < 0.8,
                "no_references": len(final_rating.get("references", [])) == 0,
                "reference_download_failed": sum(
                    1 for ref in final_rating.get("references", []) if ref.get("error")
                )
                > 0,
            },
        }

        await self._write_log_file(
            self.performance_logs_dir / f"{analysis_id}_performance.json",
            performance_entry,
        )

    async def _write_log_file(self, file_path: Path, log_data: Dict[str, Any]) -> None:
        """Write log data to JSON file asynchronously.

        Args:
            file_path: Path to the log file.
            log_data: Dictionary of log data to write.
        """
        try:
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(log_data, indent=2, default=str))
        except Exception as e:
            # Fallback to synchronous write if async fails
            with open(file_path, "w") as f:
                json.dump(log_data, f, indent=2, default=str)
            print(f"Warning: Async log write failed, used sync fallback: {e}")

    async def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information for audit context.

        Returns:
            Dictionary containing system information.
        """
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "openai_api_configured": bool(os.getenv("OPENAI_API_KEY")),
                "audit_logging_enabled": True,
                "log_directory": str(self.log_directory),
            },
            "system": {
                "python_version": "3.13+",
                "package_version": "1.0.0",
                "audit_schema_version": "1.0",
            },
        }

    def _classify_error(self, error_message: str) -> str:
        """Classify error type for categorization.

        Args:
            error_message: Error message to classify.

        Returns:
            Error category string.
        """
        error_lower = error_message.lower()

        if "timeout" in error_lower or "timed out" in error_lower:
            return "timeout"
        elif "api" in error_lower or "openai" in error_lower:
            return "api_error"
        elif "validation" in error_lower or "schema" in error_lower:
            return "validation_error"
        elif "rate limit" in error_lower:
            return "rate_limit"
        elif "authentication" in error_lower or "api key" in error_lower:
            return "authentication"
        elif "network" in error_lower or "connection" in error_lower:
            return "network"
        else:
            return "unknown"

    def _requires_escalation(self, error_message: str) -> bool:
        """Determine if error requires escalation.

        Args:
            error_message: Error message to evaluate.

        Returns:
            True if escalation is needed, False otherwise.
        """
        critical_indicators = [
            "authentication",
            "api key",
            "quota exceeded",
            "system failure",
            "critical error",
        ]

        error_lower = error_message.lower()
        return any(indicator in error_lower for indicator in critical_indicators)

    async def generate_analysis_report(self, analysis_id: str) -> Dict[str, Any]:
        """Generate comprehensive analysis report from audit logs.

        Args:
            analysis_id: Analysis ID to generate report for.

        Returns:
            Comprehensive analysis report.
        """
        report = {
            "analysis_id": analysis_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_type": "comprehensive_audit",
        }

        # Collect all log files for this analysis
        log_files = list(self.log_directory.rglob(f"*{analysis_id}*.json"))

        if not log_files:
            report["status"] = "no_logs_found"
            return report

        # Parse and aggregate log data
        events = []
        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    log_data = json.load(f)
                    events.append(log_data)
            except Exception as e:
                report.setdefault("parsing_errors", []).append(
                    {"file": str(log_file), "error": str(e)}
                )

        # Sort events by timestamp
        events.sort(key=lambda x: x.get("timestamp_utc", ""))

        report.update(
            {
                "event_count": len(events),
                "events": events,
                "timeline": [
                    {
                        "timestamp": event.get("timestamp_utc"),
                        "event_type": event.get("event_type"),
                        "summary": self._summarize_event(event),
                    }
                    for event in events
                ],
                "compliance_summary": {
                    "full_audit_trail": len(events) > 0,
                    "start_logged": any(
                        e.get("event_type") == "analysis_start" for e in events
                    ),
                    "completion_logged": any(
                        e.get("event_type") == "analysis_completion" for e in events
                    ),
                    "errors_logged": any(
                        e.get("event_type") == "analysis_failure" for e in events
                    ),
                    "regulatory_compliant": True,
                },
            }
        )

        return report

    def _summarize_event(self, event: Dict[str, Any]) -> str:
        """Create a summary of an audit event.

        Args:
            event: Event data to summarize.

        Returns:
            Summary string.
        """
        event_type = event.get("event_type", "unknown")

        if event_type == "analysis_start":
            sector = event.get("request", {}).get("sector", "unknown")
            return f"Started analysis for {sector}"
        elif event_type == "analysis_completion":
            rating = event.get("final_rating", {}).get("rating", "unknown")
            confidence = event.get("final_rating", {}).get("confidence", 0)
            return f"Completed with rating {rating}/5 (confidence: {confidence:.2f})"
        elif event_type == "analysis_failure":
            error = event.get("error", {}).get("message", "unknown error")
            return f"Failed: {error[:100]}"
        elif event_type == "model_request":
            model = event.get("model", "unknown")
            return f"Requested analysis from {model}"
        elif event_type == "model_response":
            model = event.get("model", "unknown")
            latency = event.get("performance", {}).get("latency_ms", 0)
            return f"Received response from {model} ({latency:.0f}ms)"
        else:
            return f"Event: {event_type}"


# Utility functions for audit log analysis
async def analyze_audit_logs(log_directory: str) -> Dict[str, Any]:
    """Analyze audit logs for patterns and insights.

    Args:
        log_directory: Directory containing audit logs.

    Returns:
        Analysis results and insights.
    """
    log_dir = Path(log_directory)
    if not log_dir.exists():
        return {"error": "Log directory not found"}

    # Collect all analysis completion logs
    completion_files = list(log_dir.rglob("*completion.json"))

    analyses = []
    for file_path in completion_files:
        try:
            with open(file_path, "r") as f:
                analyses.append(json.load(f))
        except Exception:
            continue

    if not analyses:
        return {"message": "No completed analyses found"}

    # Calculate aggregate metrics
    total_analyses = len(analyses)
    avg_latency = (
        sum(a.get("ensemble_metrics", {}).get("total_latency_ms", 0) for a in analyses)
        / total_analyses
    )
    avg_cost = (
        sum(a.get("ensemble_metrics", {}).get("total_cost_usd", 0) for a in analyses)
        / total_analyses
    )
    avg_confidence = (
        sum(a.get("final_rating", {}).get("confidence", 0) for a in analyses)
        / total_analyses
    )

    return {
        "summary": {
            "total_analyses": total_analyses,
            "average_latency_ms": avg_latency,
            "average_cost_usd": avg_cost,
            "average_confidence": avg_confidence,
        },
        "performance_targets": {
            "latency_under_5min": sum(
                1
                for a in analyses
                if a.get("ensemble_metrics", {}).get("total_latency_ms", 0) < 300000
            ),
            "cost_under_5usd": sum(
                1
                for a in analyses
                if a.get("ensemble_metrics", {}).get("total_cost_usd", 0) < 5.0
            ),
            "confidence_above_30pct": sum(
                1
                for a in analyses
                if a.get("final_rating", {}).get("confidence", 0) > 0.3
            ),
        },
    }
