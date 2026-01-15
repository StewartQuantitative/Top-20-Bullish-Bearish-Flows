# Options Flow Scoring Backtest

This backtesting system validates and optimizes the options flow scoring algorithm used in `top_bullish_bearish_flows.py`.

## Overview

The backtest performs:
1. **Historical Data Collection**: Fetches options flow data for multiple dates and tickers
2. **Parameter Optimization**: Grid search to find optimal weights and coefficients
3. **Correlation Analysis**: Measures relationship between flow scores and future returns
4. **Performance Testing**: Evaluates different holding periods and position sizes
5. **Visualization**: Generates comprehensive charts and graphs
6. **Reporting**: Creates detailed backtest reports with recommendations

## Files

- `backtest_flow_correlation.py` - Main backtesting script
- `backtest_results/` - Output directory for results
  - `backtest_report.txt` - Comprehensive text report
  - `backtest_results.csv` - Detailed backtest data
  - `correlation_results.csv` - Correlation metrics
  - `parameter_optimization.csv` - Parameter sensitivity analysis
  - `plots/` - All visualization charts

## Usage

### Basic Run

```bash
cd "C:\Users\willi\Documents\Python Scripts\Top bullish_bearish flows\Backtesting"
python backtest_flow_correlation.py
```

### Configuration

Edit the configuration section in `backtest_flow_correlation.py`:

```python
# Backtesting Parameters
LOOKBACK_DAYS = 60              # Number of trading days to backtest
HOLDING_PERIODS = [1, 2, 3, 5, 10]  # Days to hold positions
TOP_N_POSITIONS = [5, 10, 20]   # Number of positions to track

# Parameter Optimization Ranges
PARAM_GRID = {
    'ASK_WEIGHT': [0.8, 1.0, 1.2, 1.5],
    'BID_WEIGHT': [0.5, 0.7, 0.9, 1.0],
    'DTE_LAMBDA': [0.05, 0.10, 0.15, 0.20],
    'PRICE_CAP_ADJUSTMENT': [True, False]
}
```

## What Gets Tested

### Parameters Being Optimized

1. **ASK_WEIGHT** - Weight for aggressive trades (ask side)
   - Tests: 0.8, 1.0, 1.2, 1.5
   - Current default: 1.0

2. **BID_WEIGHT** - Weight for passive trades (bid side)
   - Tests: 0.5, 0.7, 0.9, 1.0
   - Current default: 0.7

3. **DTE_LAMBDA** - Exponential decay rate for DTE weighting
   - Tests: 0.05, 0.10, 0.15, 0.20
   - Current default: 0.10

4. **PRICE_CAP_ADJUSTMENT** - Whether to apply price/cap bias correction
   - Tests: True, False
   - Current default: True

### Strategies Tested

1. **Long Bullish** - Buy stocks with highest positive flow scores
2. **Short Bearish** - Short stocks with most negative flow scores

### Holding Periods

- 1 day (day trading)
- 2 days (short swing)
- 3 days (swing trading)
- 5 days (weekly swing)
- 10 days (bi-weekly swing)

### Position Sizes

- Top 5 positions
- Top 10 positions
- Top 20 positions

## Output Visualizations

### 1. Scatter Plots
- Flow Score vs Returns for each holding period
- Shows correlation strength and trend lines
- Files: `scatter_flow_vs_Xd_returns.png`

### 2. Correlation Heatmap
- Pearson and Spearman correlations by holding period
- Color-coded for easy interpretation
- File: `correlation_heatmap.png`

### 3. Backtest Performance
- Average returns by holding period and strategy
- Win rates across different configurations
- File: `backtest_performance.png`

### 4. Parameter Sensitivity
- How each parameter affects performance
- Identifies optimal parameter ranges
- File: `parameter_sensitivity.png`

### 5. Cumulative Returns
- Time-series performance of strategies
- Shows consistency and drawdowns
- Files: `cumulative_returns_Xd.png`

## Interpreting Results

### Correlation Metrics

- **Pearson Correlation**: Linear relationship strength
  - > 0.3: Strong positive correlation ✓
  - 0.1-0.3: Moderate correlation
  - < 0.1: Weak correlation ⚠

- **Spearman Correlation**: Monotonic relationship (rank-based)
  - More robust to outliers
  - Better for non-linear relationships

- **P-value**: Statistical significance
  - < 0.05: Statistically significant
  - < 0.01: Highly significant

### Performance Metrics

- **Average Return**: Mean return across all trades
- **Median Return**: Middle return (less affected by outliers)
- **Win Rate**: Percentage of profitable trades
- **Optimization Score**: avg_return × (win_rate / 100)

### What to Look For

✓ **Good Signs:**
- Positive correlation between flow score and returns
- Win rate > 55% for long positions
- Consistent performance across holding periods
- Parameter optimization improves results

⚠ **Warning Signs:**
- Negative or near-zero correlation
- Win rate < 50%
- High variance in returns
- Parameter changes have minimal effect

## Expected Runtime

- **Fast Mode** (30 days, 30 tickers): ~10-15 minutes
- **Standard Mode** (60 days, 60 tickers): ~30-45 minutes
- **Full Mode** (90 days, 100+ tickers): ~1-2 hours

*Runtime depends on API rate limits and network speed*

## Requirements

All requirements are already in your main project:
- massive (Massive API client)
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- python-dotenv

## Troubleshooting

### "No historical data collected"
- Check API key in `.env` file
- Verify tickers are valid
- Check date range (not too far back)
- API rate limits may be hit

### "No data with returns"
- Some tickers may not have price data for future dates
- Reduce LOOKBACK_DAYS or use more liquid tickers
- Check for market holidays

### Low correlation results
- May need more data (increase LOOKBACK_DAYS)
- Try different tickers (more volatile stocks)
- Flow scoring may need refinement
- Market conditions may not favor the strategy

## Next Steps After Backtesting

1. **Review the report** - Check `backtest_report.txt` for recommendations
2. **Analyze visualizations** - Look for patterns in the plots
3. **Update parameters** - Apply optimal parameters to main script
4. **Test forward** - Run the main script with new parameters
5. **Iterate** - Refine based on real-world results

## Notes

- Historical options data may be limited by API
- Results are based on historical data and don't guarantee future performance
- Consider transaction costs and slippage in real trading
- Backtest uses simplified bid/ask estimation
- Market conditions change - regular re-optimization recommended
