"""Main portfolio construction interface.

This module provides the primary PortfolioConstructor class that orchestrates
the entire portfolio construction pipeline from sector scores to tradeable
allocations.
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging

from .models import Portfolio, Signal, RiskMetrics, PortfolioConfig, OptimizationConfig
from .config import (
    SECTOR_LONG_ETF,
    SECTOR_INVERSE_ETF,
    CORRELATION_CLUSTERS,
)
from .signals.calibration import SignalCalibrator, ScoreMapper

logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """Main portfolio construction engine.

    Converts sector scores into beta-neutral ETF allocations with comprehensive
    risk management and optimization.

    This is the primary interface for Phase 2 portfolio construction, supporting
    the Chapter 7 notebook requirements while maintaining backward compatibility
    with the simple Phase 2 specification.
    """

    def __init__(
        self,
        config: Optional[PortfolioConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize portfolio constructor.

        Args:
            config: Portfolio configuration. If None, uses defaults.
            optimization_config: Optimization configuration. If None, uses defaults.
        """
        self.config = config or PortfolioConfig()
        self.optimization_config = optimization_config or OptimizationConfig()
        self.signal_calibrator = SignalCalibrator()
        self.score_mapper = ScoreMapper()

    def build_portfolio(
        self,
        sector_scores: Dict[str, int],
        confidence_scores: Optional[Dict[str, float]] = None,
        previous_portfolio: Optional[Portfolio] = None,
        vintage: Optional[str] = None,
    ) -> Portfolio:
        """Build portfolio from sector scores.

        Main interface for portfolio construction supporting both simple
        (Phase 2 spec) and comprehensive (Chapter 7 notebook) use cases.

        Args:
            sector_scores: Sector name -> score (1-5) mapping.
            confidence_scores: Sector name -> confidence [0,1] mapping.
            previous_portfolio: Previous portfolio for turnover analysis.
            vintage: Rebalancing vintage identifier.

        Returns:
            Complete Portfolio object with allocations and metadata.

        Raises:
            ValueError: If sector_scores is invalid or contains unknown sectors.
        """
        start_time = datetime.now()

        try:
            # Validate inputs
            self._validate_inputs(sector_scores)

            # Step 1: Convert scores to signals
            signals = self._scores_to_signals(sector_scores, confidence_scores)

            # Step 2: Map signals to ETF allocations
            long_positions, short_positions = self._signals_to_positions(signals)

            # Step 3: Apply risk constraints and target gross exposure
            long_positions, short_positions, hedge_positions = (
                self._apply_constraints_and_scale(long_positions, short_positions)
            )

            # Step 5: Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                long_positions, short_positions, hedge_positions
            )

            # Step 6: Estimate construction cost
            construction_cost = self._estimate_cost(
                long_positions, short_positions, hedge_positions, previous_portfolio
            )

            # Step 7: Create portfolio object
            portfolio = Portfolio(
                long_positions=long_positions,
                short_positions=short_positions,
                hedge_positions=hedge_positions,
                signals=signals,
                risk_metrics=risk_metrics,
                optimization_method=self.optimization_config.method,
                constraints_applied=self._get_applied_constraints(),
                construction_cost=construction_cost,
                vintage=vintage,
                config=self.config,
            )

            # Log construction summary
            construction_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Portfolio constructed in {construction_time:.2f}s: "
                f"gross={risk_metrics.gross_exposure:.3f}, "
                f"beta={risk_metrics.estimated_beta:.3f}, "
                f"max_concentration={risk_metrics.max_sector_concentration:.3f}"
            )

            return portfolio

        except Exception as e:
            logger.error(f"Portfolio construction failed: {str(e)}")
            raise

    def build_portfolio_simple(
        self,
        sector_scores: Dict[str, int],
        gross_target: float = 1.0,
        max_sector_weight: float = 0.30,
    ) -> Portfolio:
        """Simplified interface for basic portfolio construction.

        Backward-compatible interface matching the original Phase 2 specification.

        Args:
            sector_scores: Sector name -> score (1-5) mapping.
            gross_target: Target gross exposure.
            max_sector_weight: Maximum sector weight.

        Returns:
            Portfolio with basic beta-neutral construction.
        """
        # Update config for this construction
        temp_config = PortfolioConfig(
            max_sector_weight=max_sector_weight,
            max_gross_exposure=gross_target,
        )

        # Save original config and temporarily replace
        original_config = self.config
        self.config = temp_config

        try:
            portfolio = self.build_portfolio(sector_scores)
            return portfolio
        finally:
            # Restore original config
            self.config = original_config

    def _validate_inputs(self, sector_scores: Dict[str, int]) -> None:
        """Validate input sector scores.

        Args:
            sector_scores: Sector scores to validate.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not sector_scores:
            raise ValueError("sector_scores cannot be empty")

        # Check for valid sectors
        valid_sectors = set(SECTOR_LONG_ETF.keys())
        invalid_sectors = set(sector_scores.keys()) - valid_sectors
        if invalid_sectors:
            raise ValueError(f"Unknown sectors: {invalid_sectors}")

        # Check score ranges
        for sector, score in sector_scores.items():
            if not isinstance(score, int) or not (1 <= score <= 5):
                raise ValueError(f"Score for {sector} must be integer 1-5, got {score}")

    def _scores_to_signals(
        self,
        sector_scores: Dict[str, int],
        confidence_scores: Optional[Dict[str, float]] = None,
    ) -> List[Signal]:
        """Convert sector scores to calibrated signals.

        Args:
            sector_scores: Sector name -> score mapping.
            confidence_scores: Sector name -> confidence mapping.

        Returns:
            List of calibrated Signal objects.
        """
        signals = self.signal_calibrator.calibrate_scores(
            sector_scores, confidence_scores, sleeve="sector_committee"
        )

        # Store original scores in signals for consistent tilt mapping
        for signal in signals:
            signal.metadata["original_score"] = sector_scores[signal.asset]

        return signals

    def _signals_to_positions(
        self,
        signals: List[Signal],
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """Convert signals to long and short ETF positions.

        Args:
            signals: List of calibrated signals.

        Returns:
            Tuple of (long_positions, short_positions) dictionaries.
        """
        long_positions = {}
        short_positions = {}

        # Use simple score-to-tilt mapping for better control
        tilts = {}
        for signal in signals:
            # Get original score from metadata
            score = signal.metadata.get("original_score", 3)  # Default to neutral

            # Map score directly to tilt using score mapper
            score_dict = {signal.asset: score}
            tilt_dict = self.score_mapper.scores_to_tilts(score_dict)
            tilt = tilt_dict[signal.asset] * signal.confidence
            tilts[signal.asset] = tilt

        # Calculate positions ensuring target gross exposure
        total_abs_tilt = sum(abs(tilt) for tilt in tilts.values())
        if total_abs_tilt == 0:
            return long_positions, short_positions

        # Target total notional (accounting for leverage)
        target_notional = self.config.max_gross_exposure

        # First pass: calculate raw positions
        raw_long = {}
        raw_short = {}
        total_raw_notional = 0

        for sector, tilt in tilts.items():
            if tilt > 0:
                # Long position
                weight = abs(tilt)
                raw_long[sector] = weight
                total_raw_notional += weight
            elif tilt < 0:
                # Short position (adjust for leverage)
                inverse_etf, leverage = SECTOR_INVERSE_ETF[sector]
                weight = abs(tilt)
                raw_short[sector] = weight
                # Account for leverage in notional calculation
                total_raw_notional += weight

        # Scale to hit target gross exposure
        if total_raw_notional > 0:
            scale_factor = target_notional / total_raw_notional
        else:
            scale_factor = 1.0

        # Apply scaling and constraints
        for sector, weight in raw_long.items():
            scaled_weight = weight * scale_factor
            # Apply sector weight cap
            scaled_weight = min(scaled_weight, self.config.max_sector_weight)
            long_etf = SECTOR_LONG_ETF[sector]
            long_positions[long_etf] = scaled_weight

        for sector, weight in raw_short.items():
            scaled_weight = weight * scale_factor
            # Apply sector weight cap
            scaled_weight = min(scaled_weight, self.config.max_sector_weight)
            inverse_etf, leverage = SECTOR_INVERSE_ETF[sector]
            # Position size in inverse ETF (not adjusted for leverage here)
            short_positions[inverse_etf] = scaled_weight

        return long_positions, short_positions

    def _calculate_beta_hedge(
        self,
        long_positions: Dict[str, float],
        short_positions: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate beta hedge positions.

        Args:
            long_positions: Long ETF positions.
            short_positions: Short ETF positions.

        Returns:
            Hedge positions dictionary.
        """
        # More precise beta calculation
        # Assume all sector ETFs have beta ≈ 1.0 vs SPY
        long_beta_exposure = sum(long_positions.values())

        # Calculate short beta exposure considering leverage
        short_beta_exposure = 0.0
        for etf, weight in short_positions.items():
            # Find leverage for this inverse ETF
            leverage = -2.0  # Default assumption for inverse ETFs
            for sector, (inverse_etf, lev) in SECTOR_INVERSE_ETF.items():
                if inverse_etf == etf:
                    leverage = lev
                    break
            # Short positions contribute negative beta (inverse ETFs have negative beta)
            short_beta_exposure += weight * leverage  # This will be negative

        # Net beta exposure
        net_beta = long_beta_exposure + short_beta_exposure

        # Always hedge to target beta (usually 0)
        hedge_positions = {}
        target_beta = self.config.target_beta
        hedge_needed = target_beta - net_beta  # How much beta we need to add

        if abs(hedge_needed) > 0.01:  # Only hedge if meaningful exposure
            if hedge_needed > 0:
                # Need positive beta hedge - use SPY (beta ≈ +1)
                hedge_positions["SPY"] = abs(hedge_needed)
            else:
                # Need negative beta hedge - use SPDN (beta ≈ -1)
                hedge_positions["SPDN"] = abs(hedge_needed)

        return hedge_positions

    def _apply_constraints_and_scale(
        self,
        long_positions: Dict[str, float],
        short_positions: Dict[str, float],
    ) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Apply constraints and scale to target gross exposure including hedge.

        Args:
            long_positions: Long positions.
            short_positions: Short positions.

        Returns:
            Tuple of constrained and scaled (long, short, hedge) positions.
        """
        # Apply constraints first
        for positions in [long_positions, short_positions]:
            for asset in list(positions.keys()):
                max_weight = min(
                    self.config.max_asset_weight, self.config.max_sector_weight
                )
                if positions[asset] > max_weight:
                    positions[asset] = max_weight

        # Apply correlation cluster limits
        self._apply_cluster_constraints(long_positions, short_positions)

        # Apply strict sector concentration limits
        self._enforce_sector_concentration_limits(long_positions, short_positions)

        # Reserve space for hedge (20% should be enough)
        hedge_reserve = 0.20
        sector_budget = self.config.max_gross_exposure * (1.0 - hedge_reserve)

        # Scale sector positions to fit in the budget
        current_sector_total = sum(long_positions.values()) + sum(
            short_positions.values()
        )
        if current_sector_total > 0:
            sector_scale = sector_budget / current_sector_total

            for etf in long_positions:
                long_positions[etf] *= sector_scale
            for etf in short_positions:
                short_positions[etf] *= sector_scale

        # Calculate precise beta and hedge
        long_beta = sum(long_positions.values())
        short_beta = 0.0
        for etf, weight in short_positions.items():
            leverage = -2.0  # Default for inverse ETFs
            for sector, (inverse_etf, lev) in SECTOR_INVERSE_ETF.items():
                if inverse_etf == etf:
                    leverage = lev
                    break
            short_beta += weight * leverage

        sector_beta = long_beta + short_beta
        target_beta = self.config.target_beta
        hedge_needed = target_beta - sector_beta

        # Create hedge positions
        hedge_positions = {}
        if abs(hedge_needed) > 0.001:  # Only hedge if meaningful
            if hedge_needed > 0:
                # Need positive beta - use SPY
                hedge_positions["SPY"] = abs(hedge_needed)
            else:
                # Need negative beta - use SPDN
                hedge_positions["SPDN"] = abs(hedge_needed)

        # Apply concentration limits to hedge (but allow exceeding for beta control)
        for etf in hedge_positions:
            max_weight = min(
                self.config.max_asset_weight, self.config.max_sector_weight
            )
            # Only cap hedge if it's not critical for beta control
            if hedge_positions[etf] > max_weight and hedge_positions[etf] > 0.25:
                # Cap at 25% for very large hedge positions only
                hedge_positions[etf] = 0.25

        # Final scaling to hit exact gross exposure target
        total_exposure = (
            sum(long_positions.values())
            + sum(short_positions.values())
            + sum(hedge_positions.values())
        )

        if total_exposure > 0:
            final_scale = self.config.max_gross_exposure / total_exposure

            for etf in long_positions:
                long_positions[etf] *= final_scale
            for etf in short_positions:
                short_positions[etf] *= final_scale
            for etf in hedge_positions:
                hedge_positions[etf] *= final_scale

        # Final cluster constraints enforcement
        self._apply_cluster_constraints(long_positions, short_positions)

        return long_positions, short_positions, hedge_positions

    def _estimate_portfolio_beta(
        self,
        long_positions: Dict[str, float],
        short_positions: Dict[str, float],
        hedge_positions: Dict[str, float],
    ) -> float:
        """Estimate portfolio beta for constraint checking."""
        long_beta = sum(long_positions.values())

        short_beta = 0.0
        for etf, weight in short_positions.items():
            leverage = -2.0  # Default for inverse ETFs
            for sector, (inverse_etf, lev) in SECTOR_INVERSE_ETF.items():
                if inverse_etf == etf:
                    leverage = lev
                    break
            short_beta += weight * leverage

        hedge_beta = 0.0
        for etf, weight in hedge_positions.items():
            if etf == "SPDN":
                hedge_beta -= weight
            elif etf == "SPY":
                hedge_beta += weight

        return long_beta + short_beta + hedge_beta

    def _enforce_sector_concentration_limits(
        self,
        long_positions: Dict[str, float],
        short_positions: Dict[str, float],
    ) -> None:
        """Enforce strict sector concentration limits.

        Args:
            long_positions: Long positions to constrain.
            short_positions: Short positions to constrain.
        """
        # Check each sector and scale down if needed
        for sector, etf in SECTOR_LONG_ETF.items():
            # Check long position
            if (
                etf in long_positions
                and long_positions[etf] > self.config.max_sector_weight
            ):
                long_positions[etf] = self.config.max_sector_weight

            # Check short position via inverse ETF
            if sector in SECTOR_INVERSE_ETF:
                inverse_etf, leverage = SECTOR_INVERSE_ETF[sector]
                if (
                    inverse_etf in short_positions
                    and short_positions[inverse_etf] > self.config.max_sector_weight
                ):
                    short_positions[inverse_etf] = self.config.max_sector_weight

    def _apply_constraints(
        self,
        long_positions: Dict[str, float],
        short_positions: Dict[str, float],
        hedge_positions: Dict[str, float],
    ) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Apply portfolio constraints.

        Args:
            long_positions: Long positions.
            short_positions: Short positions.
            hedge_positions: Hedge positions.

        Returns:
            Tuple of constrained (long, short, hedge) positions.
        """
        # Apply concentration limits

        # Cap individual positions
        for positions in [long_positions, short_positions, hedge_positions]:
            for asset in list(positions.keys()):
                if positions[asset] > self.config.max_asset_weight:
                    positions[asset] = self.config.max_asset_weight

        # Apply correlation cluster limits
        self._apply_cluster_constraints(long_positions, short_positions)

        return long_positions, short_positions, hedge_positions

    def _apply_cluster_constraints(
        self,
        long_positions: Dict[str, float],
        short_positions: Dict[str, float],
    ) -> None:
        """Apply correlation cluster constraints.

        Args:
            long_positions: Long positions to constrain.
            short_positions: Short positions to constrain.
        """
        # Check tech cluster exposure
        tech_cluster_sectors = CORRELATION_CLUSTERS["tech_cluster"]
        tech_exposure = 0.0

        for sector in tech_cluster_sectors:
            long_etf = SECTOR_LONG_ETF[sector]
            if long_etf in long_positions:
                tech_exposure += long_positions[long_etf]

        # Scale down if exceeds cluster limit
        if tech_exposure > self.config.max_cluster_weight:
            scale_factor = self.config.max_cluster_weight / tech_exposure
            for sector in tech_cluster_sectors:
                long_etf = SECTOR_LONG_ETF[sector]
                if long_etf in long_positions:
                    long_positions[long_etf] *= scale_factor

    def _calculate_risk_metrics(
        self,
        long_positions: Dict[str, float],
        short_positions: Dict[str, float],
        hedge_positions: Dict[str, float],
    ) -> RiskMetrics:
        """Calculate portfolio risk metrics.

        Args:
            long_positions: Long positions.
            short_positions: Short positions.
            hedge_positions: Hedge positions.

        Returns:
            RiskMetrics object.
        """
        all_positions = {**long_positions, **short_positions, **hedge_positions}

        # Basic risk calculations
        gross_exposure = sum(abs(w) for w in all_positions.values())
        net_exposure = sum(long_positions.values()) - sum(short_positions.values())

        # Concentration metrics
        max_asset_concentration = (
            max(abs(w) for w in all_positions.values()) if all_positions else 0.0
        )

        # Sector concentration - calculate net sector exposure properly
        sector_exposures = {}
        for sector, etf in SECTOR_LONG_ETF.items():
            # Long exposure from sector ETF
            long_exposure = long_positions.get(etf, 0.0)

            # Short exposure from inverse ETF (if exists)
            short_exposure = 0.0
            if sector in SECTOR_INVERSE_ETF:
                inverse_etf, leverage = SECTOR_INVERSE_ETF[sector]
                # For concentration purposes, count absolute exposure
                short_exposure = short_positions.get(inverse_etf, 0.0)

            # Net sector exposure (for concentration limits)
            net_sector_exposure = max(long_exposure, short_exposure)
            sector_exposures[sector] = net_sector_exposure

        max_sector_concentration = (
            max(sector_exposures.values()) if sector_exposures else 0.0
        )

        # Estimated beta (consistent with hedge calculation)
        long_beta = sum(long_positions.values())  # Assume beta ≈ 1 for sector ETFs
        short_beta = 0.0
        for etf, weight in short_positions.items():
            # Find leverage for this inverse ETF (consistent with hedge calculation)
            leverage = -2.0  # Default assumption for inverse ETFs
            for sector, (inv_etf, lev) in SECTOR_INVERSE_ETF.items():
                if inv_etf == etf:
                    leverage = lev
                    break
            short_beta += weight * leverage

        hedge_beta = 0.0
        for etf, weight in hedge_positions.items():
            if etf == "SPDN":
                hedge_beta -= weight  # SPDN has beta ≈ -1
            elif etf == "SPY":
                hedge_beta += weight  # SPY has beta ≈ +1

        estimated_beta = long_beta + short_beta + hedge_beta

        # Tech cluster exposure
        tech_sectors = CORRELATION_CLUSTERS["tech_cluster"]
        tech_exposure = sum(
            sector_exposures.get(sector, 0.0) for sector in tech_sectors
        )

        return RiskMetrics(
            max_sector_concentration=max_sector_concentration,
            max_asset_concentration=max_asset_concentration,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            estimated_beta=estimated_beta,
            correlation_cluster_exposure=tech_exposure,
        )

    def _estimate_cost(
        self,
        long_positions: Dict[str, float],
        short_positions: Dict[str, float],
        hedge_positions: Dict[str, float],
        previous_portfolio: Optional[Portfolio] = None,
    ) -> float:
        """Estimate portfolio construction cost.

        Args:
            long_positions: Long positions.
            short_positions: Short positions.
            hedge_positions: Hedge positions.
            previous_portfolio: Previous portfolio for turnover calculation.

        Returns:
            Estimated cost in basis points.
        """
        # Simplified cost model based on position sizes and spreads
        total_cost = 0.0

        all_positions = {**long_positions, **short_positions, **hedge_positions}

        for asset, weight in all_positions.items():
            # Estimate spread cost
            if asset in SECTOR_LONG_ETF.values():
                spread_cost = 2.0  # 2bps for sector ETFs
            elif any(asset == inv_etf for inv_etf, _ in SECTOR_INVERSE_ETF.values()):
                spread_cost = 5.0  # 5bps for inverse ETFs
            else:
                spread_cost = 1.0  # 1bps for broad market ETFs

            # Cost proportional to position size
            total_cost += weight * spread_cost / 10000  # Convert bps to decimal

        return total_cost * 10000  # Return in basis points

    def _get_applied_constraints(self) -> List[str]:
        """Get list of constraints that were applied.

        Returns:
            List of constraint names.
        """
        constraints = ["sector_concentration", "asset_concentration", "gross_exposure"]

        if self.config.max_cluster_weight < 1.0:
            constraints.append("correlation_cluster")

        if abs(self.config.target_beta) < 0.5:
            constraints.append("beta_neutrality")

        return constraints
