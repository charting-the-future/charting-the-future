"""Unit tests for portfolio construction system.

Tests the core PortfolioConstructor functionality and validates Phase 2
acceptance criteria.
"""

import pytest

from sector_committee.portfolio import PortfolioConstructor, Portfolio, PortfolioConfig
from sector_committee.portfolio.config import SECTOR_LONG_ETF, SECTOR_INVERSE_ETF


class TestPortfolioConstructor:
    """Test portfolio construction functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.constructor = PortfolioConstructor()

        # Standard test sector scores
        self.test_scores = {
            "Information Technology": 5,
            "Health Care": 4,
            "Financials": 3,
            "Consumer Staples": 2,
            "Energy": 1,
            "Utilities": 2,
            "Industrials": 3,
            "Materials": 2,
            "Real Estate": 1,
            "Consumer Discretionary": 4,
            "Communication Services": 5,
        }

        # Confidence scores
        self.confidence_scores = {
            "Information Technology": 0.9,
            "Health Care": 0.8,
            "Financials": 0.7,
            "Consumer Staples": 0.6,
            "Energy": 0.8,
            "Utilities": 0.7,
            "Industrials": 0.6,
            "Materials": 0.7,
            "Real Estate": 0.8,
            "Consumer Discretionary": 0.9,
            "Communication Services": 0.85,
        }

    def test_basic_portfolio_construction(self):
        """Test basic portfolio construction from sector scores."""
        portfolio = self.constructor.build_portfolio(self.test_scores)

        # Validate basic structure
        assert isinstance(portfolio, Portfolio)
        assert len(portfolio.signals) == len(self.test_scores)
        assert portfolio.risk_metrics is not None

        # Check that we have positions
        total_positions = len(portfolio.long_positions) + len(portfolio.short_positions)
        assert total_positions > 0

        # Validate risk metrics
        risk = portfolio.risk_metrics
        assert risk.gross_exposure > 0
        assert risk.max_sector_concentration <= 0.30  # Default limit

    def test_portfolio_construction_with_confidence(self):
        """Test portfolio construction with confidence scores."""
        portfolio = self.constructor.build_portfolio(
            self.test_scores, confidence_scores=self.confidence_scores
        )

        # Check that confidence scores were applied
        assert len(portfolio.signals) == len(self.test_scores)
        for signal in portfolio.signals:
            expected_confidence = self.confidence_scores[signal.asset]
            assert signal.confidence == expected_confidence

    def test_simple_interface_backward_compatibility(self):
        """Test simple interface for backward compatibility."""
        portfolio = self.constructor.build_portfolio_simple(
            self.test_scores, gross_target=1.0, max_sector_weight=0.25
        )

        # Should produce valid portfolio
        assert isinstance(portfolio, Portfolio)
        assert portfolio.risk_metrics.max_sector_concentration <= 0.25
        assert abs(portfolio.risk_metrics.gross_exposure - 1.0) < 0.1

    def test_concentration_limits_acceptance_criteria(self):
        """Test AC2: No sector exceeding 30% allocation."""
        portfolio = self.constructor.build_portfolio(self.test_scores)

        # Should meet concentration requirement
        assert portfolio.risk_metrics.max_sector_concentration <= 0.30
        assert portfolio.risk_metrics.max_asset_concentration <= 0.30

    def test_capital_efficiency_acceptance_criteria(self):
        """Test AC2: Gross exposure = 100% ±2%."""
        portfolio = self.constructor.build_portfolio(self.test_scores)

        # Should meet capital efficiency requirement
        gross_exposure = portfolio.risk_metrics.gross_exposure
        assert 0.98 <= gross_exposure <= 1.02

    def test_performance_requirement(self):
        """Test AC3: Full portfolio construction completing within 30 seconds."""
        import time

        start_time = time.time()
        portfolio = self.constructor.build_portfolio(self.test_scores)
        construction_time = time.time() - start_time

        # Should meet performance requirement
        assert construction_time < 30.0
        assert isinstance(portfolio, Portfolio)

    def test_etf_mapping_validation(self):
        """Test AC4: All tickers verified as tradeable instruments."""
        portfolio = self.constructor.build_portfolio(self.test_scores)

        # Check that all positions use valid ETF tickers
        all_positions = portfolio.all_positions

        valid_tickers = set(SECTOR_LONG_ETF.values())
        valid_tickers.update(ticker for ticker, _ in SECTOR_INVERSE_ETF.values())
        valid_tickers.update(["SPDN", "SH", "SPY", "PSQ"])  # Hedge ETFs

        for ticker in all_positions.keys():
            assert ticker in valid_tickers, f"Invalid ticker: {ticker}"

    def test_error_handling_empty_scores(self):
        """Test error handling for empty sector scores."""
        with pytest.raises(ValueError, match="sector_scores cannot be empty"):
            self.constructor.build_portfolio({})

    def test_error_handling_invalid_sectors(self):
        """Test error handling for invalid sector names."""
        invalid_scores = {"Invalid Sector": 3}

        with pytest.raises(ValueError, match="Unknown sectors"):
            self.constructor.build_portfolio(invalid_scores)

    def test_error_handling_invalid_score_values(self):
        """Test error handling for invalid score values."""
        invalid_scores = {"Information Technology": 6}  # Outside 1-5 range

        with pytest.raises(ValueError, match="must be integer 1-5"):
            self.constructor.build_portfolio(invalid_scores)

    def test_extreme_scores_all_ones(self):
        """Test extreme scenario: all sectors rated 1 (max short)."""
        all_ones = {sector: 1 for sector in SECTOR_LONG_ETF.keys()}
        portfolio = self.constructor.build_portfolio(all_ones)

        # Should produce valid portfolio with short positions
        assert isinstance(portfolio, Portfolio)
        assert len(portfolio.short_positions) > 0
        assert portfolio.risk_metrics.net_exposure < 0  # Net short bias

    def test_extreme_scores_all_fives(self):
        """Test extreme scenario: all sectors rated 5 (max long)."""
        all_fives = {sector: 5 for sector in SECTOR_LONG_ETF.keys()}
        portfolio = self.constructor.build_portfolio(all_fives)

        # Should produce valid portfolio with long positions
        assert isinstance(portfolio, Portfolio)
        assert len(portfolio.long_positions) > 0
        assert portfolio.risk_metrics.net_exposure > 0  # Net long bias

    def test_mixed_scores_scenario(self):
        """Test mixed scores scenario."""
        mixed_scores = {
            "Information Technology": 5,
            "Energy": 1,
            "Health Care": 3,
            "Financials": 2,
            "Utilities": 4,
        }

        portfolio = self.constructor.build_portfolio(mixed_scores)

        # Should produce valid mixed portfolio
        assert isinstance(portfolio, Portfolio)
        assert len(portfolio.long_positions) > 0  # Should have long positions
        assert len(portfolio.short_positions) > 0  # Should have short positions

    def test_tech_cluster_constraint(self):
        """Test technology cluster concentration constraint."""
        # Create scores that would create high tech exposure
        tech_heavy_scores = {sector: 3 for sector in SECTOR_LONG_ETF.keys()}
        tech_heavy_scores.update(
            {
                "Information Technology": 5,
                "Communication Services": 5,
                "Consumer Discretionary": 5,
            }
        )

        portfolio = self.constructor.build_portfolio(tech_heavy_scores)

        # Tech cluster exposure should be constrained
        assert portfolio.risk_metrics.correlation_cluster_exposure <= 0.50

    def test_portfolio_metadata(self):
        """Test portfolio metadata and attribution."""
        portfolio = self.constructor.build_portfolio(self.test_scores)

        # Check metadata
        assert portfolio.optimization_method == "mvo"
        assert len(portfolio.constraints_applied) > 0
        assert portfolio.construction_cost >= 0
        assert portfolio.timestamp is not None

    def test_signal_calibration(self):
        """Test signal calibration from scores to μ forecasts."""
        portfolio = self.constructor.build_portfolio(self.test_scores)

        # Check that signals were calibrated
        signals = portfolio.signals
        assert len(signals) > 0

        for signal in signals:
            assert signal.asset in self.test_scores
            assert signal.sleeve == "sector_committee"
            assert -1.0 <= signal.mu <= 1.0  # Reasonable μ range
            assert 0.0 < signal.confidence <= 1.0

    def test_custom_configuration(self):
        """Test portfolio construction with custom configuration."""
        custom_config = PortfolioConfig(
            max_sector_weight=0.20,
            max_gross_exposure=0.80,
            target_beta=0.0,
        )

        constructor = PortfolioConstructor(config=custom_config)
        portfolio = constructor.build_portfolio(self.test_scores)

        # Should respect custom configuration
        assert portfolio.risk_metrics.max_sector_concentration <= 0.20
        assert portfolio.risk_metrics.gross_exposure <= 0.82  # Allow small tolerance


class TestScoreMapping:
    """Test score mapping functionality."""

    def test_scores_to_tilts_mapping(self):
        """Test {1,2,3,4,5} → {-2,-1,0,+1,+2} mapping."""
        constructor = PortfolioConstructor()
        mapper = constructor.score_mapper

        test_scores = {
            "Test1": 1,  # Should map to -2
            "Test2": 2,  # Should map to -1
            "Test3": 3,  # Should map to 0
            "Test4": 4,  # Should map to +1
            "Test5": 5,  # Should map to +2
        }

        tilts = mapper.scores_to_tilts(test_scores)

        assert tilts["Test1"] == -2.0
        assert tilts["Test2"] == -1.0
        assert tilts["Test3"] == 0.0
        assert tilts["Test4"] == 1.0
        assert tilts["Test5"] == 2.0

    def test_scores_to_ranks_mapping(self):
        """Test score to percentile rank mapping."""
        constructor = PortfolioConstructor()
        mapper = constructor.score_mapper

        test_scores = {
            "Test1": 1,  # Should map to 0.0
            "Test3": 3,  # Should map to 0.5
            "Test5": 5,  # Should map to 1.0
        }

        ranks = mapper.scores_to_ranks(test_scores)

        assert ranks["Test1"] == 0.0
        assert ranks["Test3"] == 0.5
        assert ranks["Test5"] == 1.0
