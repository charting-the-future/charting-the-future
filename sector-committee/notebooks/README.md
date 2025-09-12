# Chapter 6 Notebook: From Words to Trades

This directory contains the hands-on Jupyter notebook for Chapter 6 of "Charting the Future".

## Files

- `chapter_06.ipynb` - Complete hands-on implementation notebook

## Getting Started

### Prerequisites

1. **Python Environment**: Ensure you have Python 3.11+ with `uv` package manager
2. **Package Installation**: From the `sector-committee` directory, run:
   ```bash
   uv install
   ```

### Running the Notebook

1. **Navigate to sector-committee directory**:
   ```bash
   cd sector-committee
   ```

2. **Launch Jupyter Lab**:
   ```bash
   uv run jupyter lab
   ```

3. **Open the notebook**:
   - Navigate to `notebooks/chapter_06.ipynb`
   - Run cells sequentially

## Notebook Features

### 🔄 Offline Mode (Default)
- Runs without API keys using mock responses
- Perfect for learning without costs
- All functionality demonstrated with realistic data

### ⚡ Live API Mode (Optional)
- Set environment variables for OpenAI/Anthropic API keys
- Change `mode="offline"` to `mode="openai"` in adapter cells
- **Note**: Live API calls will incur charges

### 🎯 Learning Objectives Covered

1. **Two-Stage LLM Pipeline** - Deep research → structured output
2. **Tri-Pillar Methodology** - Fundamentals, sentiment, technicals  
3. **Signal Calibration** - Ratings → portfolio signals (μ)
4. **Event-Time Validation** - No look-ahead bias, realistic performance
5. **Risk Overlays** - Confidence weighting and throttling
6. **Governance Audit** - Complete regulatory compliance trails

### 📊 Exercises Included

- **Exercise A**: Adjust pillar weights and observe prompt changes
- **Exercise B**: Test different sectors and schema validation
- Interactive visualizations and performance metrics

## Integration with Production Code

The notebook directly uses your production codebase:

- `sector_committee.data_models` - Request/response structures
- `sector_committee.scoring.agents` - Main SectorAgent interface
- `sector_committee.scoring.prompts` - Tri-pillar prompt templates
- `sector_committee.scoring.llm_adapters` - Two-stage pipeline adapters
- `sector_committee.scoring.schema` - JSON validation system
- `sector_committee.scoring.audit` - Governance logging

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the sector-committee directory
cd sector-committee

# Install dependencies
uv install

# Run with correct Python path
uv run jupyter lab
```

### Matplotlib Issues
```bash
# If plots don't show, try:
%matplotlib inline
```

### API Rate Limits
- Use offline mode for learning (default)
- Set reasonable delays between API calls if using live mode

## Next Steps

After completing Chapter 6:
- **Chapter 7**: Portfolio construction and risk management
- **Production Deployment**: Replace mock functions with real implementations
- **Real Data Integration**: Connect to live market data feeds

## Support

For issues with the notebook:
1. Check that all cells run in sequence
2. Verify you're using the correct Python environment
3. Ensure the sector-committee package is properly installed
