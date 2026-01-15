# Getting Started with Backtesting

## Quick Start (Recommended)

### Step 1: Run Quick Test
Double-click `run_quick_test.bat` or run:
```bash
python quick_backtest.py
```

**What it does:**
- Tests 20 trading days with 16 liquid tickers
- Takes 5-10 minutes
- Validates that everything works
- Shows if the flow scoring has predictive power

**Output:** `quick_backtest_results/QUICK_SUMMARY.txt`

### Step 2: Review Results
Open `quick_backtest_results/QUICK_SUMMARY.txt` and check:
- ✓ Correlation > 0.15 = Good, proceed to full test
- ⚠ Correlation < 0.15 = Needs tuning

### Step 3: Run Full Backtest (if quick test looks good)
Double-click `run_full_backtest.bat` or run:
```bash
python backtest_flow_correlation.py
```

**What it does:**
- Tests 60 trading days with 60+ tickers
- Takes 30-60 minutes
- Comprehensive parameter optimization
- Detailed visualizations and reports

**Output:** `backtest_results/backtest_report.txt`

### Step 4: Advanced Analysis (optional)
```bash
python advanced_analysis.py
```

**What it does:**
- Market cap category analysis
- Volatility-based performance
- Flow magnitude insights
- Additional visualizations

**Output:** `advanced_analysis_results/`

## What to Expect

### Quick Test Output
```
quick_backtest_results/
├── QUICK_SUMMARY.txt          # Main findings
├── results.csv                # Detailed results
├── correlations.csv           # Correlation metrics
└── plots/
    ├── main_correlation.png   # Flow vs Returns scatter
    └── performance_summary.png # Strategy comparison
```

### Full Backtest Output
```
backtest_results/
├── backtest_report.txt        # Comprehensive report
├── backtest_results.csv       # All backtest data
├── correlation_results.csv    # Correlation analysis
├── parameter_optimization.csv # Parameter sensitivity
├── historical_data.json       # Raw data (for advanced analysis)
└── plots/
    ├── scatter_flow_vs_*d_returns.png
    ├── correlation_heatmap.png
    ├── backtest_performance.png
    ├── parameter_sensitivity.png
    └── cumulative_returns_*d.png
```

## Understanding the Results

### 1. Correlation Strength
- **> 0.3**: Strong - Flow scores are highly predictive ✓
- **0.15-0.3**: Moderate - Flow scores have some predictive power
- **< 0.15**: Weak - May need parameter tuning or different approach

### 2. Win Rate
- **> 55%**: Excellent for long positions
- **50-55%**: Good (profitable with positive avg returns)
- **< 50%**: Needs improvement

### 3. Average Returns
- **> 2%** for 5-day holds: Strong performance
- **1-2%** for 5-day holds: Decent performance
- **< 1%** for 5-day holds: Marginal (check transaction costs)

### 4. Sharpe Ratio
- **> 1.5**: Excellent risk-adjusted returns
- **1.0-1.5**: Good risk-adjusted returns
- **< 1.0**: Poor risk-adjusted returns

## Applying Results to Your Main Script

After finding optimal parameters, update `top_bullish_bearish_flows.py`:

```python
# OLD (lines 52-54)
ASK_WEIGHT = 1.0
BID_WEIGHT = 0.7
DTE_LAMBDA = 0.10

# NEW (use values from backtest_report.txt "OPTIMAL PARAMETERS" section)
ASK_WEIGHT = 1.2  # Example - use your actual optimal value
BID_WEIGHT = 0.9  # Example - use your actual optimal value
DTE_LAMBDA = 0.15 # Example - use your actual optimal value
```

## Troubleshooting

### "No historical data collected"
**Causes:**
- API key missing or invalid
- Network issues
- API rate limits

**Solutions:**
1. Check `.env` file has `MASSIVE_API_KEY=your_key_here`
2. Reduce number of tickers in test
3. Add delays between API calls

### "Weak correlation"
**Possible reasons:**
- Not enough data (increase LOOKBACK_DAYS)
- Market conditions (sideways market reduces signal)
- Tickers too stable (test with more volatile stocks)
- Flow scoring needs refinement

**Try:**
1. Increase LOOKBACK_DAYS to 90
2. Add more volatile tickers to TEST_TICKERS
3. Test during high-volatility market periods
4. Review parameter optimization suggestions

### "Script takes too long"
**Solutions:**
1. Reduce LOOKBACK_DAYS (try 30)
2. Reduce number of TEST_TICKERS (try 20)
3. Reduce parameter combinations in PARAM_GRID
4. Use quick_backtest.py instead

## Configuration Tips

### For Faster Testing
```python
LOOKBACK_DAYS = 20
TEST_TICKERS = ['AAPL', 'MSFT', 'TSLA', 'SPY', 'QQQ']  # 5 tickers
HOLDING_PERIODS = [5]  # Only test 5-day
```

### For Comprehensive Analysis
```python
LOOKBACK_DAYS = 90
TEST_TICKERS = [...]  # 100+ tickers
HOLDING_PERIODS = [1, 2, 3, 5, 10, 20]  # All periods
```

### For Day Trading Focus
```python
HOLDING_PERIODS = [1, 2]  # Only short-term
```

### For Swing Trading Focus
```python
HOLDING_PERIODS = [3, 5, 10]  # Medium-term
```

## Best Practices

1. **Start Small**: Always run quick_backtest.py first
2. **Check Data Quality**: Review historical_data.json to ensure good coverage
3. **Multiple Time Periods**: Test different market conditions (bull/bear/sideways)
4. **Regular Updates**: Re-run backtests quarterly to adjust parameters
5. **Forward Testing**: After optimizing, test with new data (not in backtest)
6. **Transaction Costs**: Remember to account for commissions and slippage

## Next Steps

1. ✓ Run quick test
2. ✓ Review correlation strength
3. ✓ If good, run full backtest
4. ✓ Apply optimal parameters
5. ✓ Forward test with real-time data
6. ✓ Monitor performance
7. ✓ Re-optimize periodically

## Support

If you encounter issues:
1. Check README.md for detailed documentation
2. Review error messages in console
3. Verify all requirements are installed
4. Check API key and network connectivity
5. Try reducing test scope (fewer days/tickers)

## Important Notes

⚠ **Backtesting Limitations:**
- Past performance doesn't guarantee future results
- Simplified bid/ask estimation
- No transaction costs included
- No slippage modeling
- Market conditions change

✓ **Use backtesting to:**
- Validate approach
- Optimize parameters
- Understand relationships
- Set realistic expectations
- Guide strategy refinement

**NOT to:**
- Guarantee future profits
- Replace forward testing
- Ignore market changes
- Over-optimize (curve fitting)
