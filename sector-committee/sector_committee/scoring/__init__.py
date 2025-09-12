"""Phase 1: Deep Research Scoring System.

This module implements the multi-agent sector analysis system that produces
1-5 numerical scores with comprehensive audit trails for each of the 11 SPDR
sectors.

Core Components:
- SectorAgent: Main interface for sector analysis
- SectorRating: Structured output with validation
- ModelFactory: Provider abstraction for OpenAI models
- EnsembleAggregator: Multi-model consensus scoring
- Audit logging: Complete compliance trail

Example:
    agent = SectorAgent()
    rating = await agent.analyze_sector(
        sector_name="Information Technology",
        horizon_weeks=4
    )
    assert 1 <= rating["rating"] <= 5
"""

from ..data_models import SectorRating, SectorRequest
from .agents import SectorAgent, analyze_sector_quick, analyze_all_sectors
from .factory import ModelFactory, ResearchClient
from .ensemble import EnsembleAggregator
from .schema import validate_sector_rating
from .audit import AuditLogger

__all__ = [
    "SectorRating",
    "SectorRequest",
    "SectorAgent",
    "ModelFactory",
    "ResearchClient",
    "EnsembleAggregator",
    "validate_sector_rating",
    "AuditLogger",
    "analyze_sector_quick",
    "analyze_all_sectors",
]
