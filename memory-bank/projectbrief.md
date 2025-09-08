# Project Brief: Charting the Future - LLM-Driven Quantitative Finance

## Core Mission

Build a production-ready multi-agent system for systematic investment decision-making that converts LLM agent views into beta-neutral sector portfolios. The system implements the methodology described in "Charting the Future: Harnessing LLMs for Quantitative Finance" - specifically the sector committee approach that transitions from individual agent scores to tradeable, hedged positions.

## Primary Objectives

1. **Multi-Agent Sector Committee**: Deploy specialized LLM agents (Macro, Rates, Valuation, Policy/Text) that analyze the 11 SPDR sector ETFs (XLB through XLK) and produce 1-5 ratings with structured justifications

2. **Ensemble Scoring System**: Implement tri-pillar analysis (Fundamentals, Sentiment, Technicals) with configurable weights and multi-model validation using OpenAI's deep research capabilities

3. **Portfolio Construction Pipeline**: Transform agent scores into sized, beta-neutral long/short positions using direct SPDR ETFs for longs and inverse ETFs for shorts, with appropriate leverage adjustments

4. **Production Guardrails**: Ensure auditability, cost control, capacity management, and compliance with systematic turnover controls and risk neutralization

## Key Success Criteria

- **Systematic Decision Making**: Replace discretionary sector allocation with evidence-based, auditable agent consensus
- **Risk Management**: Maintain beta neutrality and sector exposure limits while capturing agent alpha
- **Operational Excellence**: Handle transaction costs, inverse ETF tracking, and daily reset risks appropriately
- **Regulatory Compliance**: Provide complete audit trails from raw inputs through agent reasoning to final orders

## Technical Scope

The system encompasses:
- **Agent Framework**: Structured prompt templates with JSON schema enforcement
- **Research Infrastructure**: Web search integration and reference downloading for evidence-based analysis
- **Portfolio Management**: Score aggregation, tilt mapping, and beta neutralization workflows
- **Risk Controls**: Turnover budgets, concentration limits, and confidence-weighted sizing

## Investment Universe

**Core ETFs (Long Positions)**:
- 11 SPDR Sector ETFs: XLB (Materials), XLE (Energy), XLF (Financials), XLI (Industrials), XLK (Technology), XLP (Consumer Staples), XLRE (Real Estate), XLU (Utilities), XLV (Health Care), XLY (Consumer Discretionary), XLC (Communication Services)

**Inverse ETFs (Short Positions)**:
- Mix of -1x and -2x inverse funds with appropriate notional adjustments
- Broad market hedging via SPDN/SH for residual beta neutralization

## Repository Integration

This system serves as the capstone implementation for the book's companion repository, integrating with:
- Research paper summaries and academic references
- Sample datasets for backtesting and validation
- Instructor materials for educational deployment
- Documentation and governance frameworks

The project demonstrates the practical application of LLM agents in quantitative finance while maintaining institutional-grade risk management and operational discipline.
