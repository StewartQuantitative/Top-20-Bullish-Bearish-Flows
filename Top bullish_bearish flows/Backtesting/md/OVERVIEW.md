# Backtesting System Overview

## 📋 What This System Does

This comprehensive backtesting system validates and optimizes your options flow scoring algorithm by:

1. **Testing Historical Performance** - Downloads historical options flow data and measures correlation with future stock returns
2. **Optimizing Parameters** - Tests different combinations of weights and coefficients to find optimal values
3. **Validating Strategy** - Measures win rates, returns, and risk metrics across different holding periods
4. **Providing Insights** - Generates detailed reports and visualizations to guide strategy refinement

## 🎯 Why You Need This

Your main script (`top_bullish_bearish_flows.py`) uses several parameters that affect flow scoring:

```python
ASK_WEIGHT = 1.0      # How much to weight aggressive buying
BID_WEIGHT = 0.7      # How much to weight passive selling  
DTE_LAMBDA = 0.10     # How much to decay longer-dated options
```

**Questions this backtest answers:**
- ✅ Do high flow scores actually predict positive returns?
- ✅ What are the optimal parameter values?
- ✅ What holding period works best (1 day, 5 days, 10 days)?
- ✅ How many positions should you track (top 5, 10, 20)?
- ✅ Does the strategy work better for certain market caps or volatilities?

## 📁 Files in This Directory

### Main Scripts

| File | Purpose | Runtime | When to Use |
|------|---------|---------|-------------|
| `quick_backtest.py` | Fast validation test | 5-10 min | **Start here** - validates system works |
| `backtest_flow_correlation.py` | Full comprehensive backtest | 30-60 min | After quick test looks good |
| `advanced_analysis.py` | Deep-dive analysis | 2-5 min | After full backtest for extra insights |
| `apply_optimal_params.py` | Update main script | Instant | Apply optimized parameters |

### Helper Files

| File | Purpose |
|------|---------|
| `run_quick_test.bat` | Double-click to run quick test (Windows) |
| `run_full_backtest.bat` | Double-click to run full backtest (Windows) |
| `GETTING_STARTED.md` | Step-by-step guide for beginners |
| `README.md` | Detailed technical documentation |
| `requirements.txt` | Python package dependencies |

## 🚀 Quick Start (3 Steps)

### Step 1: Run Quick Test
```bash
# Windows: Double-click run_quick_test.bat
# Or from command line:
python quick_backtest.py
```

**Takes:** 5-10 minutes  
**Tests:** 20 days, 16 tickers  
**Output:** `quick_backtest_results/QUICK_SUMMARY.txt`

### Step 2: Check Results
Open `quick_backtest_results/QUICK_SUMMARY.txt` and look for:

```
Average Pearson correlation: 0.XXX
```

- **> 0.30** = 🎉 Excellent! Strong predictive power
- **0.15-0.30** = ✅ Good! Proceed to full backtest
- **< 0.15** = ⚠️ Weak signal, may need tuning

### Step 3: Apply Optimal Parameters
```bash
python apply_optimal_params.py
```

This automatically updates your main script with the best parameters found.

## 📊 What You Get

### Correlation Analysis
Measures how well flow scores predict returns:

```
Holding Period: 5 days
Pearson Correlation: 0.342 (p=0.0001)
✓ STRONG positive correlation
```

### Performance Metrics
Shows actual trading results:

```
Strategy: Long Bullish (Top 10)
Average Return: 2.8%
Win Rate: 58.3%
Sharpe Ratio: 1.42
```

### Parameter Optimization
Finds best settings:

```
OPTIMAL PARAMETERS:
  ASK_WEIGHT: 1.2 (was 1.0)
  BID_WEIGHT: 0.9 (was 0.7)
  DTE_LAMBDA: 0.15 (was 0.10)
  
Improvement: +23.5%
```

### Visualizations
Generates 8+ charts including:
- Flow Score vs Returns scatter plots
- Correlation heatmaps
- Performance by holding period
- Parameter sensitivity analysis
- Cumulative returns over time

## 🎓 Understanding Results

### Good Signs ✅
- Positive correlation (> 0.15)
- Win rate > 55%
- Consistent across holding periods
- Parameter optimization improves results
- Top flow scores outperform bottom scores

### Warning Signs ⚠️
- Negative or near-zero correlation
- Win rate < 50%
- High variance in returns
- No improvement from optimization
- Results inconsistent across periods

## 🔧 Customization

### Test Different Tickers
Edit `TEST_TICKERS` in the script:

```python
TEST_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL',  # Your tickers here
    # Add more...
]
```

### Test Different Time Periods
```python
LOOKBACK_DAYS = 60  # Change to 30, 90, etc.
HOLDING_PERIODS = [1, 2, 3, 5, 10]  # Add/remove periods
```

### Adjust Parameter Ranges
```python
PARAM_GRID = {
    'ASK_WEIGHT': [0.8, 1.0, 1.2, 1.5],  # Test your ranges
    'BID_WEIGHT': [0.5, 0.7, 0.9, 1.0],
    'DTE_LAMBDA': [0.05, 0.10, 0.15, 0.20],
}
```

## 📈 Workflow

```
1. Quick Test (5-10 min)
   ↓
2. Review Results
   ↓
3. Full Backtest (30-60 min) ← If quick test good
   ↓
4. Advanced Analysis (5 min) ← Optional deep dive
   ↓
5. Apply Parameters
   ↓
6. Forward Test with Real Data
   ↓
7. Monitor & Re-optimize Quarterly
```

## 🎯 Use Cases

### Scenario 1: First Time User
**Goal:** Validate the flow scoring works  
**Run:** `quick_backtest.py`  
**Look for:** Positive correlation > 0.15  
**Time:** 10 minutes

### Scenario 2: Parameter Tuning
**Goal:** Find optimal weights  
**Run:** `backtest_flow_correlation.py`  
**Look for:** Best parameter combination  
**Time:** 45 minutes

### Scenario 3: Strategy Refinement
**Goal:** Understand which stocks/periods work best  
**Run:** Full backtest + `advanced_analysis.py`  
**Look for:** Market cap, volatility insights  
**Time:** 60 minutes

### Scenario 4: Quarterly Re-optimization
**Goal:** Update parameters for current market  
**Run:** Full backtest → apply parameters  
**Look for:** Parameter drift from previous  
**Time:** 45 minutes

## 💡 Pro Tips

1. **Start Small** - Always run quick test first
2. **Check Data Quality** - Ensure good ticker coverage
3. **Multiple Periods** - Test bull, bear, and sideways markets
4. **Don't Over-optimize** - Avoid curve-fitting to historical data
5. **Forward Test** - Always validate with new data
6. **Regular Updates** - Re-run quarterly as markets change
7. **Transaction Costs** - Remember to account for commissions
8. **Slippage** - Real fills may differ from backtests

## ⚠️ Important Limitations

**What backtesting CAN do:**
- ✅ Validate approach has merit
- ✅ Optimize parameters
- ✅ Measure historical correlation
- ✅ Compare strategies
- ✅ Set realistic expectations

**What backtesting CANNOT do:**
- ❌ Guarantee future profits
- ❌ Account for all market conditions
- ❌ Replace real-world testing
- ❌ Predict black swan events
- ❌ Model exact execution

**Remember:** Past performance ≠ Future results

## 🆘 Troubleshooting

### Problem: "No historical data collected"
**Solution:** Check API key in `.env` file, reduce number of tickers

### Problem: "Weak correlation"
**Solution:** Try more volatile tickers, increase lookback period, test different market conditions

### Problem: "Takes too long"
**Solution:** Reduce LOOKBACK_DAYS, use fewer tickers, run quick_backtest.py instead

### Problem: "API rate limits"
**Solution:** Reduce MAX_WORKERS, add delays, spread tests over time

## 📚 Additional Resources

- **GETTING_STARTED.md** - Detailed beginner guide
- **README.md** - Technical documentation
- **Quick test output** - `quick_backtest_results/QUICK_SUMMARY.txt`
- **Full results** - `backtest_results/backtest_report.txt`
- **Advanced insights** - `advanced_analysis_results/`

## 🎬 Next Steps

1. ✅ Read this overview (you're here!)
2. ⏭️ Run `python quick_backtest.py`
3. ⏭️ Review `quick_backtest_results/QUICK_SUMMARY.txt`
4. ⏭️ If good, run `python backtest_flow_correlation.py`
5. ⏭️ Apply optimal parameters with `python apply_optimal_params.py`
6. ⏭️ Test in real-time with updated main script
7. ⏭️ Monitor results and re-optimize as needed

## 📞 Questions?

Check the documentation files:
- Quick start: `GETTING_STARTED.md`
- Technical details: `README.md`
- Parameter optimization: See `backtest_report.txt` after running

---

**Ready to start?** Run `python quick_backtest.py` or double-click `run_quick_test.bat`!
