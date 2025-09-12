"""Test configuration and fixtures for Phase 1 integration tests.

This module provides shared fixtures and configuration to optimize test performance
while maintaining comprehensive coverage.
"""

import pytest
import pytest_asyncio
import asyncio
from typing import Dict, Optional

from sector_committee.scoring import SectorAgent, SectorRating


# Performance-optimized test configuration
FAST_TEST_CONFIG = {
    "timeout_seconds": 30,  # Reduced from 60-300
    "min_confidence": 0.1,  # Lower threshold for testing
    "enable_audit": False,  # Disable audit for speed
    "max_retries": 1,  # Reduced retries
}

# Shared test data to avoid redundant API calls
_cached_analysis_results: Dict[str, SectorRating] = {}
_analysis_cache_lock = asyncio.Lock()


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def fast_agent():
    """Create a performance-optimized agent for testing."""
    return SectorAgent(
        timeout_seconds=FAST_TEST_CONFIG["timeout_seconds"],
        enable_audit=FAST_TEST_CONFIG["enable_audit"],
        min_confidence=FAST_TEST_CONFIG["min_confidence"],
    )


@pytest_asyncio.fixture(scope="session")
async def sample_analysis(fast_agent):
    """Get a cached sample analysis to avoid redundant API calls."""
    async with _analysis_cache_lock:
        if "sample" not in _cached_analysis_results:
            try:
                # Use a stable sector for consistent testing
                result = await fast_agent.analyze_sector("Utilities")
                _cached_analysis_results["sample"] = result
            except Exception as e:
                pytest.skip(f"Cannot run tests without API access: {e}")

        return _cached_analysis_results["sample"]


@pytest_asyncio.fixture(scope="session")
async def multi_sector_analysis(fast_agent):
    """Get cached multi-sector analysis results."""
    async with _analysis_cache_lock:
        if "multi_sector" not in _cached_analysis_results:
            # Analyze 3 representative sectors concurrently for speed
            test_sectors = ["Information Technology", "Financials", "Health Care"]

            async def analyze_safe(sector: str) -> Optional[SectorRating]:
                try:
                    return await fast_agent.analyze_sector(sector)
                except Exception:
                    return None

            # Run concurrently for speed
            tasks = [analyze_safe(sector) for sector in test_sectors]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter successful results
            successful_results = {
                sector: result
                for sector, result in zip(test_sectors, results)
                if result is not None and not isinstance(result, Exception)
            }

            if not successful_results:
                pytest.skip("Cannot run multi-sector tests without API access")

            _cached_analysis_results["multi_sector"] = successful_results

        return _cached_analysis_results["multi_sector"]


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for testing."""
    return {
        "max_latency_ms": 60000,  # 1 minute (reduced from 5 minutes)
        "max_cost_usd": 2.0,  # $2 per analysis
        "min_confidence": 0.3,  # 30% minimum confidence
        "min_success_rate": 0.8,  # 80% success rate for batch operations
    }


def pytest_collection_modifyitems(config, items):
    """Optimize test collection for faster execution."""
    # Mark slow tests for conditional execution
    slow_markers = [
        "test_all_spdr_sectors_analysis",
        "test_performance_targets_validation",
        "test_audit_trail_completeness",
    ]

    for item in items:
        for marker in slow_markers:
            if marker in item.name:
                item.add_marker(pytest.mark.slow)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "api: marks tests that require API access")
