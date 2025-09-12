"""Signal processing pipeline for portfolio construction.

This module provides signal calibration, blending, and confidence weighting
functionality to convert raw scores into portfolio-ready μ forecasts.

Components:
- SignalCalibrator: Rank → z → μ conversion with IC scaling
- SignalBlender: Multi-sleeve alpha blending (IC-proportional, Bayesian)
- ConfidenceWeighter: Committee agreement and stability analysis
- SignalOrthogonalizer: Factor neutralization and orthogonalization

Usage:
    from sector_committee.portfolio.signals import SignalCalibrator

    calibrator = SignalCalibrator()
    signals = calibrator.calibrate_scores(sector_scores, confidence_scores)
"""

from .calibration import SignalCalibrator, ScoreMapper

__all__ = [
    "SignalCalibrator",
    "ScoreMapper",
]
