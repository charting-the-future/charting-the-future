"""Sector Committee: Multi-Agent Quantitative Finance Analysis System.

This package implements a sophisticated multi-agent sector analysis system for
systematic investment decision-making, designed for institutional-grade
quantitative finance applications.

The system provides:
- Multi-agent sector scoring with ensemble consensus
- Structured JSON outputs with strict schema validation
- Comprehensive audit trails for regulatory compliance
- Beta-neutral portfolio construction with risk controls
- ETF-based implementation across 11 SPDR sectors

Example:
    from sector_committee.scoring import SectorAgent

    agent = SectorAgent()
    rating = await agent.analyze_sector("Information Technology")
    print(f"XLK Rating: {rating['rating']}/5")
"""

__version__ = "1.0.0"
__author__ = "Colin Alexander, CFA, CIPM"

# Core exports for Phase 1: Deep Research Scoring System
from sector_committee.scoring import (
    SectorAgent,
    SectorRating,
    SectorRequest,
    ModelFactory,
    EnsembleAggregator,
)

__all__ = [
    "SectorAgent",
    "SectorRating",
    "SectorRequest",
    "ModelFactory",
    "EnsembleAggregator",
]
