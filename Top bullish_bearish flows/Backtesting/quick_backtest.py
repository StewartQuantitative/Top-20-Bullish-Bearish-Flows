"""
Quick Backtest - Faster version with reduced scope for rapid testing
Run this first to validate the system before running the full backtest
"""

import os
import sys
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Fix potential stderr encoding issues that can cause "I/O operation on closed file" errors
try:
    if sys.stderr and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    if sys.stdout and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
except Exception:
    pass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment from parent Python Scripts directory (3 levels up)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(env_path)
API_KEY = os.getenv("MASSIVE_API_KEY")
if not API_KEY:
    print(f"ERROR: MASSIVE_API_KEY not found in .env file!")
    print(f"  Looked for .env at: {env_path}")
    exit(1)

from massive import RESTClient
massive_client = RESTClient(API_KEY)

# ============================================================================
# QUICK TEST CONFIGURATION - Reduced scope for fast validation
# ============================================================================

# Use same configuration as full backtest but with reduced scope
# For cross-sectional IC: Need 20+ dates with no overlap
# Math: 30 dates × 10 day spacing = 300 days minimum
ROLLING_WINDOW_DAYS = 450       # 15 months (need enough dates to find 30 with sufficient volatility)
NUM_RANDOM_DATES = 30           # Always gets exactly 30 dates (keeps trying until requirement met)
MIN_DATE_SPACING = 10           # 10 days between dates (prevents overlap with 7-day holding)
MIN_VOLATILITY = 0.0            # No volatility filter - test all market conditions
                                # Script will keep searching until it finds 30 dates
HOLDING_PERIODS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]    # Testing 0-10 day holding periods
# Testing shorter-term predictive power of options flow

# Overfitting prevention (adjusted for quick backtest)
TRAIN_TEST_SPLIT = 0.60         # 60% for optimization, 40% for validation (larger test set)
OUT_OF_SAMPLE_DAYS = 0          # No additional holdout - use 60/40 split only

# High Volume Options Mode: Select tickers by options volume (NO LOOK-AHEAD BIAS)
USE_HIGH_VOLUME_OPTIONS = True  # True = dynamic selection by volume, False = fixed ticker list
NUM_TICKERS_PER_DATE = 40       # Top 40 tickers by options volume per date
BALANCE_FLOW = True             # True = balance bullish/bearish flow for diverse signals

# Ticker Selection Strategy:
# - Uses ALL tickers from stocks_with_significant_options_flow.csv (filtered by open interest)
# - These are pre-filtered for significant options activity (open interest >= 150k)
# - Fetches in batches until NUM_TICKERS_PER_DATE meet volume requirements
# - Minimum volume threshold: 200 contracts (lowered to ensure sufficient tickers for all dates)

# TEST_TICKERS: Only used if USE_HIGH_VOLUME_OPTIONS = False
TEST_TICKERS = []  # Not used in high volume mode

# Parameter ranges for Bayesian optimization (matches full backtest)
# NOTE: Centered around live script defaults from top_bullish_bearish_flows.py
# ASK_WEIGHT is fixed at 1.0 (not optimized, not in params), PRICE_CAP_ADJUSTMENT fixed to True
PARAM_RANGES = {
    'BID_WEIGHT': (0.4, 0.9),           # Expanded lower: 0.5 → 0.4 (optimal was at boundary)
    'DTE_LAMBDA': (0.03, 0.20),         # Expanded both: 0.05-0.15 → 0.03-0.20 (optimal near lower)
    'SECONDARY_CLASS_WEIGHT': (0.20, 0.40),  # Keep as-is (optimal in middle)
    'MIN_SECONDARY_THRESHOLD': (0.15, 0.30)  # Expanded upper: 0.25 → 0.30 (optimal at boundary)
}

N_TRIALS = 50   # Reduced for faster iteration (30 dates × 30 tickers = 900 data points is sufficient)

# Use same centralized historical data directory as main backtest
HISTORICAL_DATA_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "historical_data")
HISTORICAL_DATA_DIR = os.path.join(HISTORICAL_DATA_BASE, "json_files")  # JSON cache files
OUTPUT_DIR = "quick_backtest_results"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

# Ensure historical data directories exist
os.makedirs(HISTORICAL_DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(HISTORICAL_DATA_BASE, "flat_files_cache"), exist_ok=True)
os.makedirs(os.path.join(HISTORICAL_DATA_BASE, "aggregated_flat_files_cache"), exist_ok=True)

# DeepSeek Configuration (same as full backtest)
USE_LOCAL_DEEPSEEK = os.getenv("USE_LOCAL_DEEPSEEK", "true").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")

# ============================================================================
# Import functions from main backtest
# ============================================================================

# Import all necessary functions from the main backtest script
import backtest_flow_correlation as main_backtest

# Share class groups (same as full backtest)
SHARE_CLASS_GROUPS = main_backtest.SHARE_CLASS_GROUPS

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("QUICK BACKTEST - Rapid Validation")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Configuration:")
    print(f"  Rolling window: {ROLLING_WINDOW_DAYS} days")
    print(f"  Random dates: {NUM_RANDOM_DATES} sampled")
    if USE_HIGH_VOLUME_OPTIONS:
        print(f"  Selection: Top {NUM_TICKERS_PER_DATE} by options volume per date")
        print(f"  Balance: {'Enabled' if BALANCE_FLOW else 'Disabled'} (bullish/bearish diversity)")
    else:
        print(f"  Tickers: {len(TEST_TICKERS)} (fixed list)")
    print(f"  Holding periods: {HOLDING_PERIODS} (swing trading)")
    print(f"  Optimization trials: {N_TRIALS}")
    print(f"  Methodology: IC-based optimization (quant standard cross-sectional IC)")
    print(f"  Estimated runtime: 5-10 minutes\n")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Step 1: Generate random trading dates (same logic as full backtest)
    start_date = datetime.now().date()
    days_back = 0
    while days_back < ROLLING_WINDOW_DAYS:
        if start_date.weekday() < 5:
            days_back += 1
        if days_back < ROLLING_WINDOW_DAYS:
            start_date -= timedelta(days=1)
    while start_date.weekday() >= 5:
        start_date -= timedelta(days=1)
    
    min_spacing = max(HOLDING_PERIODS) + 1
    trading_dates = main_backtest.get_random_trading_dates_in_window(
        start_date, ROLLING_WINDOW_DAYS, NUM_RANDOM_DATES, min_spacing, MIN_VOLATILITY
    )
    print(f"Random trading dates: {len(trading_dates)} days sampled from {ROLLING_WINDOW_DAYS}-day window")
    print(f"Minimum spacing: {min_spacing} days")
    print(f"Date range: {trading_dates[0]} to {trading_dates[-1]}\n")
    
    # Step 1.5: Validate flat files have volume data (CRITICAL CHECK)
    # Check the first date to ensure volume data exists
    if trading_dates:
        try:
            from flat_files_loader import validate_flat_files_have_volume
            validate_flat_files_have_volume(trading_dates[0].strftime('%Y-%m-%d'))
        except ValueError as e:
            print(f"\n{'='*80}")
            print("BACKTEST ABORTED")
            print(f"{'='*80}")
            print(str(e))
            print(f"\nCannot proceed with backtesting without volume data.")
            return
        except Exception as e:
            print(f"\n{'='*80}")
            print("ERROR VALIDATING FLAT FILES")
            print(f"{'='*80}")
            print(f"Error: {e}")
            print(f"\nCannot proceed with backtesting.")
            return
    
    # Step 2: Collect historical data
    if USE_HIGH_VOLUME_OPTIONS:
        historical_data = main_backtest.collect_historical_data_high_volume(
            trading_dates, 
            max_workers=20,
            num_tickers=NUM_TICKERS_PER_DATE,
            balance_flow=BALANCE_FLOW
        )
    else:
        historical_data = main_backtest.collect_historical_data(TEST_TICKERS, trading_dates)
    
    if not historical_data:
        print("ERROR: No historical data collected!")
        return
    
    # Save raw data
    raw_data_path = os.path.join(OUTPUT_DIR, 'historical_data.json')
    with open(raw_data_path, 'w') as f:
        json.dump(historical_data, f, indent=2)
    print(f"[OK] Raw data saved to {raw_data_path}")
    
    # Step 3: Add future returns
    data_with_returns = main_backtest.add_future_returns(historical_data, HOLDING_PERIODS)
    
    if not data_with_returns:
        print("ERROR: No data with returns!")
        return
    
    # Step 4: Optimize parameters (using Bayesian optimization like full backtest)
    # Temporarily override N_TRIALS for quick test
    original_trials = main_backtest.N_TRIALS
    original_ranges = main_backtest.PARAM_RANGES
    main_backtest.N_TRIALS = N_TRIALS
    main_backtest.PARAM_RANGES = PARAM_RANGES
    
    best_params, param_results = main_backtest.optimize_parameters(
        data_with_returns, HOLDING_PERIODS
    )
    
    # Restore original values
    main_backtest.N_TRIALS = original_trials
    main_backtest.PARAM_RANGES = original_ranges
    
    print(f"\n{'='*80}")
    print("OPTIMAL PARAMETERS FOUND:")
    print(f"{'='*80}")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    # Step 5: Calculate final metrics with optimal parameters
    print(f"\n{'='*80}")
    print("CALCULATING FINAL METRICS")
    print(f"{'='*80}")
    
    # Test with normalization enabled
    main_backtest.calculate_flow_scores_for_data(data_with_returns, best_params, 
                                                  use_cross_sectional_norm=True)
    
    # Primary: IC-based metrics (quant standard) - WITH normalization
    correlation_results_with_norm = main_backtest.calculate_correlation_metrics(
        data_with_returns, HOLDING_PERIODS, use_normalized_scores=True
    )
    
    # Test WITHOUT normalization for comparison
    main_backtest.calculate_flow_scores_for_data(data_with_returns, best_params, 
                                                  use_cross_sectional_norm=False)
    
    correlation_results_without_norm = main_backtest.calculate_correlation_metrics(
        data_with_returns, HOLDING_PERIODS, use_normalized_scores=False
    )
    
    # Use normalized results for main analysis (standard approach)
    correlation_results = correlation_results_with_norm
    
    # Secondary: Ranking metrics for validation
    ranking_results = main_backtest.calculate_ranking_metrics(data_with_returns, HOLDING_PERIODS, top_pct=0.10)
    
    # Print IC summary with normalization comparison
    if not correlation_results.empty:
        print("\nIC Results (Cross-sectional per date) - WITH Normalization:")
        print(correlation_results.to_string(index=False))
        
        if not correlation_results_without_norm.empty:
            print("\nIC Results (Cross-sectional per date) - WITHOUT Normalization:")
            print(correlation_results_without_norm.to_string(index=False))
            
            # Compare ICs
            print("\n" + "="*80)
            print("NORMALIZATION COMPARISON")
            print("="*80)
            for period in HOLDING_PERIODS:
                with_norm = correlation_results[correlation_results['holding_period'] == period]
                without_norm = correlation_results_without_norm[correlation_results_without_norm['holding_period'] == period]
                if not with_norm.empty and not without_norm.empty:
                    ic_with = with_norm['ic'].iloc[0]
                    ic_without = without_norm['ic'].iloc[0]
                    diff = ic_with - ic_without
                    print(f"  {period}d: With={ic_with:.4f}, Without={ic_without:.4f}, Diff={diff:+.4f}")
        
        # Pearson vs Spearman comparison
        print("\n" + "="*80)
        print("PEARSON vs SPEARMAN CORRELATION COMPARISON")
        print("="*80)
        for period in HOLDING_PERIODS:
            period_data = correlation_results[correlation_results['holding_period'] == period]
            if not period_data.empty:
                pearson = period_data['pearson_correlation'].iloc[0]
                spearman = period_data['spearman_correlation'].iloc[0]
                ic = period_data['ic'].iloc[0]
                diff = abs(pearson - spearman)
                print(f"  {period}d: Pearson={pearson:.4f}, Spearman={spearman:.4f}, IC={ic:.4f}, |Diff|={diff:.4f}")
        
        swing_ic = correlation_results[correlation_results['holding_period'].isin([p for p in HOLDING_PERIODS if 3 <= p <= 7])]
        if not swing_ic.empty:
            avg_ic = swing_ic['ic'].mean()
            avg_ic_tstat = swing_ic['ic_tstat'].mean()
            print(f"\nAverage IC (3-7 day): {avg_ic:.4f}")
            print(f"Average IC t-statistic: {avg_ic_tstat:.2f}")
    
    # Regime analysis
    print("\n" + "="*80)
    print("REGIME-BASED IC ANALYSIS")
    print("="*80)
    regime_results = main_backtest.calculate_ic_by_regime(data_with_returns, HOLDING_PERIODS)
    for regime in ['low_vol', 'normal', 'high_vol']:
        if regime_results.get(regime) is not None:
            regime_df = regime_results[regime]
            if not regime_df.empty:
                swing_regime = regime_df[regime_df['holding_period'].isin([p for p in HOLDING_PERIODS if 3 <= p <= 7])]
                if not swing_regime.empty:
                    avg_ic = swing_regime['ic'].mean()
                    sample_size = swing_regime['sample_size'].iloc[0] if 'sample_size' in swing_regime.columns else 0
                    print(f"\n{regime.upper()} Regime:")
                    print(f"  Average IC (3-7 day): {avg_ic:.4f}")
                    print(f"  Sample size: {sample_size}")
                    print(regime_df[['holding_period', 'ic', 'ic_tstat', 'sample_size']].to_string(index=False))
        else:
            print(f"\n{regime.upper()} Regime: Insufficient data")
    
    # Print ranking summary
    if not ranking_results.empty:
        print("\nRanking Results (Top 10% vs Bottom 10%):")
        print(ranking_results.to_string(index=False))
        swing_ranking = ranking_results[ranking_results['holding_period'].isin([p for p in HOLDING_PERIODS if 3 <= p <= 7])]
        if not swing_ranking.empty:
            avg_spread = swing_ranking['avg_spread'].mean()
            avg_win_rate = swing_ranking['win_rate'].mean()
            print(f"\nAverage Spread: {avg_spread:.4f}%")
            print(f"Average Win Rate: {avg_win_rate:.1f}%")
    
    # Step 6: Export results to CSV
    print(f"\n{'='*80}")
    print("EXPORTING RESULTS TO CSV")
    print(f"{'='*80}")
    
    main_backtest.export_results_to_csv(
        correlation_results, best_params, param_results, data_with_returns, OUTPUT_DIR
    )
    
    # Save ranking results
    if not ranking_results.empty:
        ranking_results.to_csv(os.path.join(OUTPUT_DIR, 'ranking_results.csv'), index=False)
        print(f"[OK] Ranking results saved to {OUTPUT_DIR}/ranking_results.csv")
    
    # Step 7: Generate visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    main_backtest.create_visualizations(
        data_with_returns, correlation_results, param_results, PLOT_DIR
    )
    
    # Step 8: Generate report
    print(f"\n{'='*80}")
    print("GENERATING REPORT")
    print(f"{'='*80}")
    
    main_backtest.generate_report(
        data_with_returns, correlation_results, best_params, param_results, OUTPUT_DIR
    )
    
    print(f"\n{'='*80}")
    print("QUICK BACKTEST COMPLETE!")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Plots saved to: {PLOT_DIR}")
    print(f"\nNext step: Run full backtest for comprehensive analysis")


if __name__ == "__main__":
    main()
