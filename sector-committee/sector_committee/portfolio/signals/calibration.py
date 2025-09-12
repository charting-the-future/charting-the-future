"""Signal calibration: converting raw scores to μ forecasts.

This module implements the calibration pipeline that converts heterogeneous
signals (e.g., 1-5 scores) into commensurate μ forecasts suitable for
portfolio optimization.

Key features:
- Rank → z-score → μ conversion
- IC-based scaling with half-life decay
- Confidence weighting integration
- Signal stability analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import logging

from ..models import Signal

logger = logging.getLogger(__name__)


@dataclass
class CalibrationParams:
    """Parameters for signal calibration.

    Attributes:
        ic_estimate: Information coefficient estimate for scaling.
        half_life: Signal decay half-life in days.
        volatility_target: Target volatility for μ scaling.
        confidence_floor: Minimum confidence threshold.
        max_z_score: Maximum z-score cap (for outlier control).
        regime_adjustment: Whether to adjust for market regimes.
    """

    ic_estimate: float = 0.05
    half_life: int = 20
    volatility_target: float = 0.16  # 16% annualized
    confidence_floor: float = 0.1
    max_z_score: float = 2.5
    regime_adjustment: bool = True


class ScoreMapper:
    """Maps discrete scores to normalized values.

    Converts integer scores (e.g., 1-5) to normalized values suitable
    for further processing in the calibration pipeline.
    """

    def __init__(self, score_range: Tuple[int, int] = (1, 5)):
        """Initialize score mapper.

        Args:
            score_range: (min_score, max_score) tuple defining valid range.
        """
        self.min_score, self.max_score = score_range
        self.score_count = self.max_score - self.min_score + 1

    def scores_to_ranks(self, scores: Dict[str, int]) -> Dict[str, float]:
        """Convert scores to percentile ranks.

        Args:
            scores: Asset -> score mapping.

        Returns:
            Asset -> percentile rank [0, 1] mapping.
        """
        if not scores:
            return {}

        # Convert to normalized ranks [0, 1]
        ranks = {}
        for asset, score in scores.items():
            if not (self.min_score <= score <= self.max_score):
                logger.warning(
                    f"Score {score} for {asset} outside range {self.score_range}"
                )
                score = max(self.min_score, min(score, self.max_score))

            # Map score to percentile rank
            rank = (score - self.min_score) / (self.max_score - self.min_score)
            ranks[asset] = rank

        return ranks

    def scores_to_tilts(self, scores: Dict[str, int]) -> Dict[str, float]:
        """Convert scores to symmetric tilts around zero.

        Maps scores to tilts in range [-2, +2] for portfolio construction.

        Args:
            scores: Asset -> score mapping.

        Returns:
            Asset -> tilt [-2, +2] mapping.
        """
        tilts = {}
        for asset, score in scores.items():
            if not (self.min_score <= score <= self.max_score):
                logger.warning(
                    f"Score {score} for {asset} outside range {self.score_range}"
                )
                score = max(self.min_score, min(score, self.max_score))

            # Map {1,2,3,4,5} → {-2,-1,0,+1,+2}
            tilt = (score - 3.0) * (2.0 / 2.0)  # Center on 3, scale to ±2
            tilts[asset] = tilt

        return tilts


class SignalCalibrator:
    """Calibrates raw signals into portfolio-ready μ forecasts.

    Implements the full calibration pipeline from raw scores to μ values
    suitable for portfolio optimization.
    """

    def __init__(self, params: Optional[CalibrationParams] = None):
        """Initialize signal calibrator.

        Args:
            params: Calibration parameters. If None, uses defaults.
        """
        self.params = params or CalibrationParams()
        self.score_mapper = ScoreMapper()

    def calibrate_scores(
        self,
        scores: Dict[str, int],
        confidence_scores: Optional[Dict[str, float]] = None,
        sleeve: str = "default",
    ) -> List[Signal]:
        """Calibrate raw scores into Signal objects.

        Args:
            scores: Asset -> raw score mapping.
            confidence_scores: Asset -> confidence [0,1] mapping.
            sleeve: Signal sleeve identifier.

        Returns:
            List of calibrated Signal objects.
        """
        if not scores:
            return []

        # Step 1: Convert scores to ranks
        ranks = self.score_mapper.scores_to_ranks(scores)

        # Step 2: Convert ranks to z-scores
        z_scores = self._ranks_to_z_scores(ranks)

        # Step 3: Scale z-scores to μ forecasts
        mu_forecasts = self._z_scores_to_mu(z_scores)

        # Step 4: Apply confidence weighting
        if confidence_scores:
            mu_forecasts = self._apply_confidence_weighting(
                mu_forecasts, confidence_scores
            )

        # Step 5: Create Signal objects
        signals = []
        for asset in scores:
            confidence = confidence_scores.get(asset, 1.0) if confidence_scores else 1.0
            confidence = max(confidence, self.params.confidence_floor)

            signal = Signal(
                asset=asset,
                mu=mu_forecasts[asset],
                confidence=confidence,
                half_life=self.params.half_life,
                ic_estimate=self.params.ic_estimate,
                sleeve=sleeve,
            )
            signals.append(signal)

        return signals

    def _ranks_to_z_scores(self, ranks: Dict[str, float]) -> Dict[str, float]:
        """Convert percentile ranks to z-scores.

        Args:
            ranks: Asset -> percentile rank [0,1] mapping.

        Returns:
            Asset -> z-score mapping.
        """
        z_scores = {}
        for asset, rank in ranks.items():
            # Convert rank to z-score using inverse normal CDF
            # Clip to avoid extreme values
            rank_clipped = max(0.001, min(0.999, rank))
            z_score = stats.norm.ppf(rank_clipped)

            # Apply z-score cap
            z_score = max(
                -self.params.max_z_score, min(z_score, self.params.max_z_score)
            )
            z_scores[asset] = z_score

        return z_scores

    def _z_scores_to_mu(self, z_scores: Dict[str, float]) -> Dict[str, float]:
        """Convert z-scores to μ forecasts.

        Scales z-scores using IC estimate and volatility target to produce
        expected return forecasts.

        Args:
            z_scores: Asset -> z-score mapping.

        Returns:
            Asset -> μ forecast mapping.
        """
        mu_forecasts = {}

        # Scale factor: IC * volatility_target
        scale_factor = self.params.ic_estimate * self.params.volatility_target

        for asset, z_score in z_scores.items():
            mu = z_score * scale_factor
            mu_forecasts[asset] = mu

        return mu_forecasts

    def _apply_confidence_weighting(
        self,
        mu_forecasts: Dict[str, float],
        confidence_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply confidence weighting to μ forecasts.

        Args:
            mu_forecasts: Asset -> μ forecast mapping.
            confidence_scores: Asset -> confidence [0,1] mapping.

        Returns:
            Asset -> confidence-weighted μ forecast mapping.
        """
        weighted_mu = {}
        for asset, mu in mu_forecasts.items():
            confidence = confidence_scores.get(asset, 1.0)
            confidence = max(confidence, self.params.confidence_floor)
            weighted_mu[asset] = mu * confidence

        return weighted_mu

    def estimate_signal_ic(
        self,
        historical_signals: List[Dict[str, float]],
        historical_returns: List[Dict[str, float]],
        lookback_periods: int = 252,
    ) -> float:
        """Estimate information coefficient from historical data.

        Args:
            historical_signals: Historical signal values by period.
            historical_returns: Historical forward returns by period.
            lookback_periods: Number of periods to use for estimation.

        Returns:
            Estimated information coefficient.
        """
        if len(historical_signals) < 2 or len(historical_returns) < 2:
            return self.params.ic_estimate

        # Use most recent periods up to lookback limit
        n_periods = min(
            len(historical_signals), len(historical_returns), lookback_periods
        )

        correlations = []
        for i in range(n_periods):
            if i >= len(historical_signals) or i >= len(historical_returns):
                continue

            signals = historical_signals[i]
            returns = historical_returns[i]

            # Find common assets
            common_assets = set(signals.keys()) & set(returns.keys())
            if len(common_assets) < 3:  # Need minimum assets for meaningful correlation
                continue

            signal_values = [signals[asset] for asset in common_assets]
            return_values = [returns[asset] for asset in common_assets]

            # Calculate correlation
            if len(signal_values) >= 3:
                corr, _ = stats.pearsonr(signal_values, return_values)
                if not np.isnan(corr):
                    correlations.append(corr)

        if correlations:
            # Return median correlation as IC estimate
            return np.median(correlations)
        else:
            return self.params.ic_estimate

    def adjust_for_regime(
        self,
        signals: List[Signal],
        regime_state: str = "normal",
    ) -> List[Signal]:
        """Adjust signals based on market regime.

        Args:
            signals: List of signals to adjust.
            regime_state: Current market regime ("stress", "normal", "calm").

        Returns:
            List of regime-adjusted signals.
        """
        if not self.params.regime_adjustment:
            return signals

        # Regime adjustment factors
        regime_factors = {
            "stress": 0.7,  # Reduce signal strength in stress
            "normal": 1.0,  # No adjustment
            "calm": 1.2,  # Increase signal strength in calm markets
        }

        factor = regime_factors.get(regime_state, 1.0)

        adjusted_signals = []
        for signal in signals:
            adjusted_signal = Signal(
                asset=signal.asset,
                mu=signal.mu * factor,
                confidence=signal.confidence,
                half_life=signal.half_life,
                ic_estimate=signal.ic_estimate,
                sleeve=signal.sleeve,
                timestamp=signal.timestamp,
                metadata={**signal.metadata, "regime_adjustment": factor},
            )
            adjusted_signals.append(adjusted_signal)

        return adjusted_signals
