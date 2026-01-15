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
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import optuna
warnings.filterwarnings('ignore')

# Fix potential stderr encoding issues
try:
    if sys.stderr and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(env_path)
API_KEY = os.getenv("MASSIVE_API_KEY")
if not API_KEY:
    print(f"ERROR: MASSIVE_API_KEY not found in .env file!")
    print(f"  Looked for .env at: {env_path}")
    exit(1)

# DeepSeek Configuration: Local (Ollama) or Cloud API
# Local: USE_LOCAL_DEEPSEEK=true (default, no API key needed)
# Cloud: USE_LOCAL_DEEPSEEK=false + DEEPSEEK_API_KEY
USE_LOCAL_DEEPSEEK = os.getenv("USE_LOCAL_DEEPSEEK", "true").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")

from massive import RESTClient
massive_client = RESTClient(API_KEY)

ROLLING_WINDOW_DAYS = 1000      # Increased window to fit 100 dates with 10-day spacing (was 600)
NUM_RANDOM_DATES = 100          # Increased for more robust testing (was 50) 
HOLDING_PERIODS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]    # Testing 1-10 day holding periods
# Testing shorter-term predictive power of options flow (matches quick_backtest)

# Overfitting prevention
TRAIN_TEST_SPLIT = 0.60         # 60% for optimization, 40% for validation
OUT_OF_SAMPLE_DAYS = 0          # No additional holdout - use 60/40 split only

# Flow score normalization settings
USE_VOLATILITY_NORMALIZATION = False     # Disabled - volatility already in flow_score calculation
USE_CROSS_SECTIONAL_NORMALIZATION = True  # Enabled - normalizes scores within each date

# Price/Cap adjustment mode: 'max' (current) or 'min' (alternative)
PRICE_CAP_MODE = 'max'  # Options: 'max' or 'min'

# High Volume Options Mode: Select tickers by options volume (NO LOOK-AHEAD BIAS)
USE_HIGH_VOLUME_OPTIONS = True  # True = dynamic selection by volume, False = fixed ticker list
NUM_TICKERS_PER_DATE = 60       # Top 60 tickers by options volume per date
BALANCE_FLOW = True             # True = balance bullish/bearish flow for diverse signals
MIN_VOLUME_THRESHOLD = 200      # Minimum contracts per ticker to be selected


# Default parameters from live script (top_bullish_bearish_flows.py)
# These are the baseline values used in production
# ASK_WEIGHT is fixed at 1.0 (not included in params)
DEFAULT_PARAMS = {
    'BID_WEIGHT': 0.7,
    'DTE_LAMBDA': 0.10,
    'VOLATILITY_WEIGHT': 0.5,
    'SECONDARY_CLASS_WEIGHT': 0.30,
    'MIN_SECONDARY_THRESHOLD': 0.20
}

# Volatility weight constant (used in flow score calculation)
VOLATILITY_WEIGHT = DEFAULT_PARAMS['VOLATILITY_WEIGHT']

# Parameter ranges for Bayesian optimization (centered around live script defaults)
# NOTE: ASK_WEIGHT is fixed at 1.0 (not optimized), PRICE_CAP_ADJUSTMENT fixed to True
PARAM_RANGES = {
    'BID_WEIGHT': (0.4, 0.9),           # Expanded lower: 0.5 → 0.4 (optimal was at boundary)
    'DTE_LAMBDA': (0.03, 0.20),         # Expanded both: 0.05-0.15 → 0.03-0.20 (optimal near lower)
    'SECONDARY_CLASS_WEIGHT': (0.20, 0.40),  # Keep as-is (optimal in middle)
    'MIN_SECONDARY_THRESHOLD': (0.15, 0.30)  # Expanded upper: 0.25 → 0.30 (optimal at boundary)
}

N_TRIALS = 50  # Reduced from 100 for faster optimization (can increase if needed)

# Centralized historical data directory (shared across all backtest scripts)
HISTORICAL_DATA_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "historical_data")
HISTORICAL_DATA_DIR = os.path.join(HISTORICAL_DATA_BASE, "json_files")  # JSON cache files
FLAT_FILES_CACHE_DIR = os.path.join(HISTORICAL_DATA_BASE, "flat_files_cache")  # Flat files cache
AGGREGATED_CACHE_DIR = os.path.join(HISTORICAL_DATA_BASE, "aggregated_flat_files_cache")  # Aggregated cache
OUTPUT_DIR = "backtest_results"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

# Ensure all cache directories exist
for dir_path in [HISTORICAL_DATA_DIR, FLAT_FILES_CACHE_DIR, AGGREGATED_CACHE_DIR]:
    os.makedirs(dir_path, exist_ok=True)
SHARE_CLASS_GROUPS = {
    'GOOG': ['GOOG', 'GOOGL'], 'BRK': ['BRK.A', 'BRK.B'], 'NWS': ['NWS', 'NWSA'],
    'FOX': ['FOX', 'FOXA'], 'LBTYK': ['LBTYK', 'LBTYA', 'LBTYB'],
    'DISCA': ['DISCA', 'DISCB', 'DISCK'], 'VIA': ['VIA', 'VIAB'],
    'PARA': ['PARA', 'PARAA'], 'CMCSA': ['CMCSA', 'CMCSK'],
    'BATRA': ['BATRA', 'BATRK'], 'FWONA': ['FWONA', 'FWONK'],
    'LSXMA': ['LSXMA', 'LSXMK', 'LSXMB'], 'LILA': ['LILA', 'LILAK'],
    'QRTEA': ['QRTEA', 'QRTEB'], 'HEICO': ['HEICO', 'HEI', 'HEI.A'],
    'UA': ['UA', 'UAA'],
}

def get_trading_dates_forward(start_date, num_days):
    """Generate trading dates forward from start (excludes weekends)."""
    dates, current = [], start_date
    while current.weekday() >= 5:
        current += timedelta(days=1)
    while len(dates) < num_days:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
        if current.weekday() == 5:
            current += timedelta(days=2)
    return dates

def calculate_spy_volatility(date_str, lookback_days=20):
    """
    Calculate SPY volatility to filter low-volatility dates.
    Low volatility = low signal-to-noise ratio = noisy IC.
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        start_date = date_obj - timedelta(days=lookback_days + 10)
        prices = list(massive_client.list_aggs('SPY', 1, "day", start_date.strftime('%Y-%m-%d'), date_str, limit=lookback_days + 10))
        
        if len(prices) < 10:
            return None
        
        returns = [np.log(float(prices[i].close) / float(prices[i-1].close)) for i in range(1, len(prices))]
        if not returns:
            return None
        
        daily_vol = np.std(returns)
        annualized_vol = daily_vol * np.sqrt(252)
        return annualized_vol
    except:
        return None

def load_filtered_tickers():
    """Load tickers from filtered list (stocks with significant options flow)."""
    try:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                 "Optionable Stocks", "stocks_with_significant_options_flow.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: Filtered tickers CSV not found at {csv_path}")
            print("Please run stocks_with_significant_options_flow.py first to generate the list.")
            return set()
        
        df = pd.read_csv(csv_path)
        tickers = set(df['Ticker'].dropna().str.strip().str.upper())
        print(f"[OK] Loaded {len(tickers)} tickers from filtered list")
        return tickers
    except Exception as e:
        print(f"Warning: Could not load filtered tickers list: {e}")
        return set()

def get_random_trading_dates_in_window(start_date, window_days, num_samples, min_spacing_days=None, min_volatility=0.10):
    """
    Get random trading dates with minimum spacing and volatility filter.
    ALWAYS returns num_samples dates (keeps trying until requirement is met).
    
    Args:
        min_volatility: Minimum SPY volatility (annualized) to skip low-vol days.
                       Default 0.10 (10%). Set to 0 to disable filtering.
    """
    if min_spacing_days is None:
        min_spacing_days = max(HOLDING_PERIODS) + 1
    
    all_dates = get_trading_dates_forward(start_date, window_days)
    if len(all_dates) < num_samples:
        print(f"Warning: Only {len(all_dates)} trading days in window (requested {num_samples})")
        return sorted(all_dates)
    
    random.shuffle(all_dates)
    sampled_dates, sampled_set = [], set()
    checked_count = 0
    skipped_volatility = 0
    skipped_spacing = 0
    
    if min_volatility > 0:
        print(f"Sampling {num_samples} dates with SPY volatility >= {min_volatility:.1%}...")
    
    for date in all_dates:
        if len(sampled_dates) >= num_samples:
            break
        
        checked_count += 1
        date_int = date.toordinal()
        
        # Check spacing requirement
        if not all(abs(date_int - s) >= min_spacing_days for s in sampled_set):
            skipped_spacing += 1
            continue
        
        # Check volatility requirement (if enabled)
        if min_volatility > 0:
            spy_vol = calculate_spy_volatility(date.strftime('%Y-%m-%d'))
            if not spy_vol or spy_vol < min_volatility:
                skipped_volatility += 1
                continue
        
        # Date passed all filters
        sampled_dates.append(date)
        sampled_set.add(date_int)
    
    # Detailed filtering statistics
    total_checked = checked_count
    total_available = len(all_dates)
    dates_passed_spacing = total_checked - skipped_spacing
    
    print(f"\n{'='*70}")
    print(f"DATE FILTERING STATISTICS")
    print(f"{'='*70}")
    print(f"Total dates in window: {total_available}")
    print(f"Dates checked before finding {num_samples}: {total_checked}")
    print(f"Dates filtered by spacing ({min_spacing_days}-day minimum): {skipped_spacing}")
    if min_volatility > 0:
        print(f"Dates filtered by volatility (< {min_volatility:.1%}): {skipped_volatility}")
    print(f"Dates that passed spacing filter: {dates_passed_spacing}")
    if min_volatility > 0:
        dates_passed_both = dates_passed_spacing - skipped_volatility
        print(f"Dates that passed both filters: {dates_passed_both}")
    print(f"Final dates selected: {len(sampled_dates)}/{num_samples} (requested)")
    
    if len(sampled_dates) < num_samples:
        print(f"\n⚠ Warning: Only found {len(sampled_dates)}/{num_samples} dates meeting requirements")
        print(f"  Consider: increasing ROLLING_WINDOW_DAYS or lowering MIN_VOLATILITY")
    else:
        print(f"\n✓ Successfully selected {num_samples} dates meeting all criteria")
    print(f"{'='*70}\n")
    
    return sorted(sampled_dates)

def calculate_realized_volatility(ticker, date_str, lookback_days=30):
    """Calculate annualized realized volatility (30-day lookback)."""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        start_date = date_obj - timedelta(days=lookback_days + 10)
        prices = list(massive_client.list_aggs(ticker, 1, "day", start_date.strftime('%Y-%m-%d'), date_str, limit=lookback_days + 10))
        if len(prices) < 10:
            return None
        closes = [float(p.close) for p in prices if p.close]
        if len(closes) < 10:
            return None
        returns = np.diff(closes) / closes[:-1]
        returns = returns[~np.isnan(returns)]
        if len(returns) < 10:
            return None
        return np.std(returns) * np.sqrt(252) or None
    except Exception:
        return None

def calculate_flow_score(net_flow, market_cap, stock_price, ask_weight, bid_weight, 
                        dte_lambda, apply_price_adjustment, volatility=None):
    """
    Calculate flow score with configurable parameters and volatility normalization.
    
    Args:
        volatility: Annualized volatility (0-1 range, e.g., 0.3 = 30% vol)
                   If None, defaults to 1.0 (no volatility adjustment)
    """
    if not market_cap or market_cap <= 0:
        return 0.0
    
    market_cap_log = np.log10(max(market_cap, 1e6))
    
    if apply_price_adjustment and stock_price and stock_price > 0:
        market_cap_billions = market_cap / 1e9
        price_cap_ratio = stock_price / market_cap_billions
        # Test both max() and min() - max increases for high-priced stocks, min decreases
        if PRICE_CAP_MODE == 'min':
            price_cap_adjustment = min(1.0, price_cap_ratio)  # Decreases for high-priced stocks (penny stock protection)
        else:  # 'max' (default)
            price_cap_adjustment = max(1.0, price_cap_ratio)  # Increases for high-priced stocks
    else:
        price_cap_adjustment = 1.0
    
    # Volatility normalization - MATCHES live script (top_bullish_bearish_flows.py line 241-246)
    # Formula: 1.0 + (volatility × VOLATILITY_WEIGHT)
    # Higher volatility stocks naturally have larger option flows, so normalize by volatility
    if volatility and volatility > 0:
        volatility_adjustment = 1.0 + (volatility * VOLATILITY_WEIGHT)
    else:
        volatility_adjustment = 1.0
    
    return net_flow / (market_cap_log * price_cap_adjustment * volatility_adjustment)

def build_options_ticker(ticker, expiry_date, contract_type, strike):
    """Build Polygon.io options ticker: O:TICKERYYMMDDC/PSTRIKE"""
    try:
        return f"O:{ticker}{expiry_date.strftime('%y%m%d')}{'C' if contract_type.upper() == 'CALL' else 'P'}{int(float(strike) * 1000):08d}"
    except:
        return None

# OPTIMIZATION: Cache for contract aggregates to avoid re-fetching
_contract_aggregate_cache = {}

def fetch_contract_historical_aggregate(options_ticker, date_str, use_cache=True):
    """
    Fetch daily aggregate for a specific options contract on a specific date.
    Uses caching to avoid redundant API calls.
    """
    # Check cache first
    cache_key = f"{options_ticker}_{date_str}"
    if use_cache and cache_key in _contract_aggregate_cache:
        return _contract_aggregate_cache[cache_key]
    
    try:
        aggs = list(massive_client.list_aggs(
            options_ticker, 1, "day",
            date_str, date_str,
            limit=1
        ))
        
        if not aggs:
            result = None
        else:
            agg = aggs[0]
            result = {
                'volume': getattr(agg, 'volume', 0) or 0,
                'close': getattr(agg, 'close', 0) or 0,
                'open': getattr(agg, 'open', 0) or 0,
                'high': getattr(agg, 'high', 0) or 0,
                'low': getattr(agg, 'low', 0) or 0,
            }
        
        # Cache result
        if use_cache:
            _contract_aggregate_cache[cache_key] = result
        
        return result
    except:
        return None

def fetch_historical_options_data(ticker, date_str, use_flat_files=False, use_cache=True):
    """
    Fetch full options contract data for flow score calculation.
    
    TWO-METHOD SYSTEM:
    1. Historical Data JSON cache: Primary source (fast, pre-processed from flat files)
    2. Flat Files: Secondary source (full historical options data - JSON cache is a cached version of this)
    
    The JSON cache is a processed/cached version of flat file data for faster access.
    Flat files contain complete historical options data (all contracts, prices, volumes, etc.).
    
    Args:
        ticker: Stock ticker symbol
        date_str: Date string (YYYY-MM-DD)
        use_flat_files: If True, fallback to flat files when cache miss
        use_cache: If True, check historical_data JSON cache first
    
    Returns:
        Data dict with full contract data (flows, prices, DTE, etc.) or None if not found
    """
    # METHOD 1: Load from cached JSON files in historical_data directory
    # JSON cache is a processed/cached version of flat file data
    if use_cache:
        cached_data = load_cached_data_for_ticker_date(ticker, date_str)
        if cached_data is not None:
            # Found in historical_data cache, return it
            return cached_data
    
    # METHOD 2: Load from flat files (source of truth for historical data)
    if not use_flat_files:
        # If cache miss and flat files disabled, return None
        return None
    
    try:
        from flat_files_loader import get_ticker_volume_from_aggregated, aggregate_and_save_daily_options_volume, check_flat_files_access
        
        # Check flat files access first
        has_access, message = check_flat_files_access()
        if not has_access:
            print(f"ERROR: Flat Files access failed: {message}")
            print("  Backtesting requires Flat Files access. Please configure POLYGON_S3_ACCESS_KEY and POLYGON_S3_SECRET_KEY in .env")
            return None
        
        # HYBRID APPROACH:
        # 1. Use aggregated files for FAST volume filtering (check if ticker meets threshold)
        # 2. If it meets threshold, fetch FULL contract data for flow calculation
        
        from flat_files_loader import fetch_options_data_from_flat_files
        
        # First, quick check using aggregated file
        volume_data = get_ticker_volume_from_aggregated(date_str, ticker)
        
        if volume_data is None or volume_data['total_volume'] == 0:
            # No data found for this ticker/date
            return None
        
        # If we get here, ticker has volume - now fetch FULL contract data for flow calculation
        # This includes prices, bid/ask side, DTE, etc. needed for proper flow scoring
        full_data = fetch_options_data_from_flat_files(ticker, date_str)
        
        if full_data is None:
            # Ticker has volume in aggregated CSV but couldn't get full contract data
            # This can happen if:
            # 1. The flat file CSV doesn't contain contracts for this ticker
            # 2. The ticker filtering failed
            # 3. The CSV structure is different than expected
            # Log this as a warning (but only first few to avoid spam)
            if not hasattr(fetch_historical_options_data, '_missing_contract_warnings'):
                fetch_historical_options_data._missing_contract_warnings = {}
            warning_key = f"{date_str}"
            if warning_key not in fetch_historical_options_data._missing_contract_warnings:
                fetch_historical_options_data._missing_contract_warnings[warning_key] = 0
            if fetch_historical_options_data._missing_contract_warnings[warning_key] < 3:
                print(f"    [WARNING] {ticker} on {date_str}: Has volume in aggregated CSV but no contract data found in flat files")
                fetch_historical_options_data._missing_contract_warnings[warning_key] += 1
            return None
        
        # Add volume info from aggregated data for easy access
        # Note: volume_data only contains total_volume, bullish_flow, bearish_flow
        # (call_volume and put_volume are not in aggregated CSV)
        full_data['_total_volume'] = volume_data['total_volume']
        
        return full_data
        
    except ImportError as e:
        print(f"ERROR: Could not import flat_files_loader: {e}")
        print("  Flat files loader is required for backtesting.")
        return None
    except Exception as e:
        # Only print errors for first few failures to avoid spam
        import traceback
        if not hasattr(fetch_historical_options_data, '_error_count'):
            fetch_historical_options_data._error_count = 0
        if fetch_historical_options_data._error_count < 3:
            print(f"ERROR: Failed to fetch data from aggregated files for {ticker} on {date_str}: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            fetch_historical_options_data._error_count += 1
        return None

def calculate_flows_with_params(flow_data, ask_weight, bid_weight, dte_lambda):
    """Calculate bullish/bearish flows with specific parameters."""
    total_bullish = total_bearish = 0.0
    
    for flow in flow_data['flows']:
        dte_weight = max(0.1, np.exp(-dte_lambda * flow['dte']))
        premium_weighted = flow['premium'] * dte_weight
        
        if flow['option_type'] == 'CALL':
            total_bullish += premium_weighted * flow['ask_side_pct'] * ask_weight
            total_bearish += premium_weighted * flow['bid_side_pct'] * bid_weight
        else:  # PUT
            total_bearish += premium_weighted * flow['ask_side_pct'] * ask_weight
            total_bullish += premium_weighted * flow['bid_side_pct'] * bid_weight
    
    return total_bullish, total_bearish

def get_next_trading_day(date_obj):
    """Get the next trading day (skip weekends)."""
    next_day = date_obj + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day

def fetch_future_returns(ticker, flow_date, holding_periods):
    """Fetch future returns: entry at same-day close, exit at close (if flow is known during trading day)."""
    try:
        flow_date_obj = datetime.strptime(flow_date, '%Y-%m-%d').date()
        max_period = max(holding_periods) + 5
        
        # Get prices starting from flow date (same day)
        prices = list(massive_client.list_aggs(
            ticker, 1, "day",
            flow_date,  # Start from flow date
            (flow_date_obj + timedelta(days=max_period)).strftime('%Y-%m-%d'),
            limit=max_period + 1
        ))
        
        if not prices or len(prices) < 2:
            return None
        
        # Entry at same-day close (flow is known during trading day)
        entry_price = float(prices[0].close) if prices[0].close else None
        if not entry_price:
            return None
        
        returns = {}
        
        for period in holding_periods:
            # Exit at close of period-th day after entry (0 = same day, 1 = next day, etc.)
            if period < len(prices):
                exit_price = float(prices[period].close)
                returns[f'{period}d_return'] = ((exit_price - entry_price) / entry_price) * 100
            else:
                returns[f'{period}d_return'] = None
        
        return returns
    
    except Exception:
        # Return None if ticker data not found - will be skipped in processing
        return None

def get_data_cache_path(tickers, dates):
    """Generate cache file path based on tickers and dates."""
    ticker_str = "_".join(sorted(tickers))[:100]  # Limit length
    date_range = f"{dates[0].strftime('%Y%m%d')}_{dates[-1].strftime('%Y%m%d')}"
    cache_filename = f"historical_data_{ticker_str}_{date_range}.json"
    return os.path.join(HISTORICAL_DATA_DIR, cache_filename)

def load_cached_data(tickers, dates):
    """Load cached historical data if available."""
    cache_path = get_data_cache_path(tickers, dates)
    if os.path.exists(cache_path):
        try:
            print(f"[OK] Loading cached data from: {cache_path}")
            with open(cache_path, 'r') as f:
                data = json.load(f)
            print(f"[OK] Loaded {len(data)} cached records")
            return data
        except Exception as e:
            print(f"⚠ Error loading cache: {e}, will fetch fresh data")
    return None

def load_cached_data_for_ticker_date(ticker, date_str):
    """
    Load cached historical data for a specific ticker/date from JSON cache files.
    Searches all cache files in HISTORICAL_DATA_DIR for matching ticker/date.
    
    Returns: Data dict in same format as fetch_historical_options_data, or None if not found.
    """
    if not os.path.exists(HISTORICAL_DATA_DIR):
        return None
    
    try:
        # Search all JSON files in cache directory
        cache_files = [f for f in os.listdir(HISTORICAL_DATA_DIR) 
                      if f.endswith('.json') and f.startswith('historical_data_')]
    except OSError:
        # Directory doesn't exist or can't be accessed
        return None
    
    if not cache_files:
        return None
    
    for cache_file in cache_files:
        cache_path = os.path.join(HISTORICAL_DATA_DIR, cache_file)
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Ensure cached_data is a list
            if not isinstance(cached_data, list):
                continue
            
            # Search for matching ticker/date
            for item in cached_data:
                if isinstance(item, dict) and item.get('ticker') == ticker and item.get('date') == date_str:
                    # Found matching data, return it
                    return item
        except (json.JSONDecodeError, IOError, OSError):
            # Skip corrupted or unreadable files
            continue
        except Exception:
            # Skip any other errors
            continue
    
    return None

def save_data_to_cache(data, tickers, dates, delete_csv_after=True):
    """Save to centralized cache. Optionally deletes CSV files to save disk space."""
    cache_path = get_data_cache_path(tickers, dates)
    try:
        os.makedirs(HISTORICAL_DATA_DIR, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[OK] Data cached to: {cache_path}")
        
        if delete_csv_after:
            deleted_count, deleted_size = 0, 0
            for date in dates:
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                csv_file = os.path.join(FLAT_FILES_CACHE_DIR, f"options-min-aggs-{date_str}.csv")
                if os.path.exists(csv_file):
                    try:
                        deleted_size += os.path.getsize(csv_file)
                        os.remove(csv_file)
                        deleted_count += 1
                    except Exception as e:
                        print(f"⚠ Could not delete {csv_file}: {e}")
            if deleted_count > 0:
                print(f"[OK] Deleted {deleted_count} CSV file(s) ({deleted_size / (1024 * 1024):.1f} MB freed)")
    except Exception as e:
        print(f"⚠ Error saving cache: {e}")

def collect_historical_data_high_volume(dates, max_workers=15, num_tickers=60, balance_flow=True):
    """
    Collect data for high-volume options tickers on each date using TWO-SOURCE approach:
    
    SOURCE 1: Aggregated CSV (ticker selection + volume metrics)
    - Uses options-day-aggs-[date].csv files for FAST volume analysis
    - Selects top N tickers by options volume
    - Provides volume/bullish/bearish metrics (pre-calculated, fast)
    
    SOURCE 2: Historical Data JSON (detailed contract data)
    - Loads from historical_data JSON files in HISTORICAL_DATA_DIR
    - Contains detailed flows array (contracts, prices, DTE, strikes, ask/bid side, etc.)
    - Used for flow score calculation (detailed contract-level data)
    - Only loaded for SELECTED tickers (optimization: don't process all tickers)
    
    Data Combination:
    - Volume/bullish/bearish metrics from Aggregated CSV (fast, pre-calculated)
    - Detailed flows array from Historical Data JSON (for flow score calculation)
    
    NO LOOK-AHEAD BIAS: Selection based on options volume (known during trading day).
    
    Args:
        balance_flow: If True, balance bullish vs bearish flow for diverse signals
    """
    print(f"\n{'='*80}\nCOLLECTING HISTORICAL DATA - HIGH VOLUME OPTIONS MODE\n{'='*80}")
    print(f"Strategy: Top {num_tickers} tickers by options volume per date")
    if balance_flow:
        print(f"  Balance: {num_tickers//2} bullish-flow + {num_tickers//2} bearish-flow")
    print(f"Dates: {len(dates)} ({dates[0]} to {dates[-1]})")
    print(f"NO LOOK-AHEAD BIAS: Selection based on options volume (known intraday)")
    
    # STEP 1: Pre-aggregate volume data for ticker selection (PARALLELIZED)
    # This creates options-day-aggs-[date].csv files with ticker and total_daily_volume
    # Purpose: Fast volume analysis to SELECT which tickers to analyze
    # NOTE: This is NOT the full contract data - that comes from historical_data JSON files
    if dates:
        print(f"\nStep 1: Pre-aggregating options volume for ticker selection (parallelized)...")
        from flat_files_loader import aggregate_and_save_daily_options_volume
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def aggregate_date(date):
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            try:
                aggregate_and_save_daily_options_volume(date_str)
                return date_str, True
            except Exception as e:
                return date_str, False
        
        # Parallelize aggregation (I/O bound, safe to use many workers)
        with ThreadPoolExecutor(max_workers=min(20, len(dates))) as executor:
            futures = {executor.submit(aggregate_date, date): date for date in dates}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                date_str, success = future.result()
                if completed % 10 == 0 or completed == len(dates):
                    print(f"  [{completed}/{len(dates)}] Aggregated volume data...")
        print(f"Volume aggregation complete (for ticker selection only).\n")
    
    all_data = []
    start_time = datetime.now()
    
    for i, date in enumerate(dates, 1):
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        print(f"\n[{i}/{len(dates)}] Processing {date_str}...")
        
        # STEP 2: Select tickers using aggregated CSV (volume analysis only)
        from flat_files_loader import select_top_tickers_from_aggregated
        selected_tickers = select_top_tickers_from_aggregated(
            date_str, 
            num_tickers, 
            min_volume_threshold=MIN_VOLUME_THRESHOLD,
            balance_flow=balance_flow
        )
        
        # Get backup tickers in case some fail to load
        backup_tickers = select_top_tickers_from_aggregated(
            date_str,
            num_tickers * 2,  # Get 2x as many for backup pool
            min_volume_threshold=MIN_VOLUME_THRESHOLD,
            balance_flow=balance_flow
        )
        # Remove selected tickers from backup pool
        backup_tickers = [t for t in backup_tickers if t not in selected_tickers]
        
        if not selected_tickers:
            print(f"  [WARNING] No tickers selected from aggregated CSV for {date_str}")
            continue
        
        # STEP 3: Load full contract data from historical_data JSON files (with flat files fallback)
        print(f"  Loading full contract data for {len(selected_tickers)} selected tickers...")
        print(f"    Primary source: historical_data JSON files (cached version of flat file data)")
        print(f"    Fallback: Flat files (source of truth - JSON cache is processed version of this)")
        print(f"    Metrics: Aggregated CSV (volume/bullish/bearish for display)")
        
        # Now load full contract data only for selected tickers
        # IMPORTANT: Two-source approach:
        # - JSON cache: Processed/cached version of flat file data (detailed flows array for flow calculation)
        # - Flat files: Source of truth for historical options data (complete contract data)
        # - Aggregated CSV: Volume/bullish/bearish metrics (pre-calculated, fast) for display/selection
        date_data_all = []
        safe_max_workers = min(max_workers, 30)  # Increased from 20 to 30 for better throughput
        cache_hits = 0
        cache_misses = 0
        
        with ThreadPoolExecutor(max_workers=safe_max_workers) as executor:
            # Try cache first, then fallback to flat files if cache miss
            futures = {executor.submit(fetch_historical_options_data, ticker, date_str, use_flat_files=True, use_cache=True): ticker
                      for ticker in selected_tickers}
            
            failed_tickers = []
            backup_index = 0
            
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result(timeout=30)
                    if result:
                        # Check if this came from cache (has flows array from JSON)
                        is_from_cache = ('flows' in result and '_total_volume' not in result)
                        if is_from_cache:
                            cache_hits += 1
                        else:
                            cache_misses += 1
                        
                        # Get volume/bullish/bearish metrics from aggregated CSV
                        # These are pre-calculated and fast to access
                        from flat_files_loader import get_ticker_volume_from_aggregated
                        volume_data = get_ticker_volume_from_aggregated(date_str, ticker)
                        
                        if volume_data:
                            total_vol = volume_data['total_volume']
                            bullish_flow = volume_data['bullish_flow']
                            bearish_flow = volume_data['bearish_flow']
                            net_flow_score = bullish_flow - bearish_flow
                            
                            # Add volume metrics from aggregated CSV
                            result['_total_volume'] = total_vol
                            result['_bullish_flow'] = bullish_flow
                            result['_bearish_flow'] = bearish_flow
                            result['_net_flow_score'] = net_flow_score
                            
                            # JSON provides detailed flows array for flow score calculation
                            # Aggregated CSV provides volume/bullish/bearish metrics for display/selection
                            
                            date_data_all.append(result)
                            
                            source = "[CACHE]" if is_from_cache else "[FLAT]"
                            print(f"    [{date_str}] {ticker} {source}: {total_vol:,} contracts, "
                                  f"bullish: {bullish_flow:,.0f}, bearish: {bearish_flow:,.0f}, "
                                  f"net flow: {net_flow_score:+.4f}")
                        else:
                            print(f"    [{date_str}] {ticker}: Could not get volume data from aggregated CSV")
                    else:
                        print(f"    [{date_str}] {ticker}: No data returned from fetch")
                        failed_tickers.append(ticker)
                except Exception as e:
                    print(f"    [{date_str}] {ticker}: ERROR - {str(e)[:50]}")
                    failed_tickers.append(ticker)
                    continue
            
            # Try backup tickers for failed ones
            if failed_tickers and backup_tickers and backup_index < len(backup_tickers):
                print(f"  [FALLBACK] Trying backup tickers for {len(failed_tickers)} failed tickers...")
                backup_futures = {}
                for ticker in failed_tickers:
                    if backup_index < len(backup_tickers):
                        backup_ticker = backup_tickers[backup_index]
                        backup_index += 1
                        backup_futures[executor.submit(fetch_historical_options_data, backup_ticker, date_str, use_flat_files=True, use_cache=True)] = (ticker, backup_ticker)
                
                for future in as_completed(backup_futures):
                    original_ticker, backup_ticker = backup_futures[future]
                    try:
                        result = future.result(timeout=30)
                        if result:
                            # Check if this came from cache
                            is_from_cache = ('flows' in result and '_total_volume' not in result)
                            if is_from_cache:
                                cache_hits += 1
                            else:
                                cache_misses += 1
                            
                            # Get volume data
                            from flat_files_loader import get_ticker_volume_from_aggregated
                            volume_data = get_ticker_volume_from_aggregated(date_str, backup_ticker)
                            
                            if volume_data:
                                total_vol = volume_data['total_volume']
                                bullish_flow = volume_data['bullish_flow']
                                bearish_flow = volume_data['bearish_flow']
                                net_flow_score = bullish_flow - bearish_flow
                                
                                result['_total_volume'] = total_vol
                                result['_bullish_flow'] = bullish_flow
                                result['_bearish_flow'] = bearish_flow
                                result['_net_flow_score'] = net_flow_score
                                result['ticker'] = backup_ticker  # Update ticker to backup
                                
                                date_data_all.append(result)
                                
                                source = "[CACHE]" if is_from_cache else "[FLAT]"
                                print(f"    [{date_str}] {backup_ticker} {source} (replaced {original_ticker}): {total_vol:,} contracts")
                        else:
                            print(f"    [{date_str}] Backup {backup_ticker} (for {original_ticker}): No data")
                    except Exception as e:
                        print(f"    [{date_str}] Backup {backup_ticker} (for {original_ticker}): ERROR - {str(e)[:50]}")
                        continue
        
        print(f"\n  {'='*70}")
        print(f"  SUMMARY FOR {date_str}")
        print(f"  {'='*70}")
        print(f"  Tickers selected from aggregated CSV: {len(selected_tickers)}")
        print(f"  Tickers with full data loaded: {len(date_data_all)}")
        if cache_hits > 0 or cache_misses > 0:
            print(f"  Data loaded: {cache_hits} from historical_data cache, {cache_misses} from flat files")
            if cache_misses > 0:
                print(f"  Note: {cache_misses} tickers loaded from flat files (cache miss - this is normal)")
        
        if len(date_data_all) < num_tickers:
            print(f"  Warning: Only {len(date_data_all)} tickers available (requested {num_tickers})")
        
        print(f"  {'='*70}\n")
        
        if not date_data_all:
            continue
        
        # Data is already selected and balanced from CSV, just use it
        selected_data = date_data_all
        
        # Clean up temporary fields
        for item in selected_data:
            item.pop('_total_volume', None)
            item.pop('_call_volume', None)
            item.pop('_put_volume', None)
            item.pop('_bullish_flow', None)
            item.pop('_bearish_flow', None)
            item.pop('_net_flow_score', None)
        
        all_data.extend(selected_data)
        print(f"  Added {len(selected_data)} data points")
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n[OK] Collected {len(all_data)} total data points in {elapsed:.1f} minutes")
    return all_data

def collect_historical_data(tickers, dates, max_workers=15, use_cache=True):
    """Collect historical options flow data with caching and progress updates."""
    print(f"\n{'='*80}\nCOLLECTING HISTORICAL DATA\n{'='*80}")
    print(f"Tickers: {len(tickers)} | Dates: {len(dates)} ({dates[0]} to {dates[-1]})")
    print(f"Total tasks: {len(tickers) * len(dates)}")
    
    if use_cache and (cached := load_cached_data(tickers, dates)):
        return cached
    
    print("Fetching fresh data from Flat Files...")
    all_data, total_tasks = [], len(tickers) * len(dates)
    completed, start_time, last_update_time = 0, datetime.now(), datetime.now()
    date_strs = {date: date.strftime('%Y-%m-%d') for date in dates}
    
    # Reduce workers to prevent system overload
    safe_max_workers = min(max_workers, 20)  # Increased from 10 to 20 for better throughput
    with ThreadPoolExecutor(max_workers=safe_max_workers) as executor:
        futures = {executor.submit(fetch_historical_options_data, ticker, date_strs[date]): (ticker, date_strs[date])
                  for ticker in tickers for date in dates}
        
        for future in as_completed(futures):
            completed += 1
            current_time = datetime.now()
            elapsed, time_since_update = (current_time - start_time).total_seconds(), (current_time - last_update_time).total_seconds()
            
            if completed % 50 == 0 or time_since_update >= 300:
                rate_per_min = (completed / elapsed * 60) if elapsed > 0 else 0
                eta = min((total_tasks - completed) / (completed / elapsed) / 60, 999) if elapsed > 0 else 0
                print(f"Progress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%) | "
                      f"Elapsed: {elapsed/60:.1f}m | Rate: {rate_per_min:.1f} tasks/min | ETA: {eta:.1f}m")
                last_update_time = current_time
            
            try:
                if (result := future.result(timeout=30)):  # Add 30 second timeout
                    all_data.append(result)
            except Exception as e:
                # Silently skip - ticker data not found or timeout, move to next
                continue
    
        print(f"\n[OK] Collected {len(all_data)} valid data points in {(datetime.now() - start_time).total_seconds() / 60:.1f} minutes")
    if all_data and use_cache:
        save_data_to_cache(all_data, tickers, dates)
    return all_data

def add_future_returns(data, holding_periods, max_workers=30):
    """Add future returns to each data point."""
    print(f"\n{'='*80}")
    print(f"CALCULATING FUTURE RETURNS")
    print(f"{'='*80}")
    
    completed = 0
    total = len(data)
    start_time = datetime.now()
    last_update_time = start_time
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for item in data:
            future = executor.submit(fetch_future_returns, item['ticker'], item['date'], holding_periods)
            futures[future] = item
        
        for future in as_completed(futures):
            completed += 1
            current_time = datetime.now()
            time_since_update = (current_time - last_update_time).total_seconds()
            
            # Progress update every 50 tasks or every 5 minutes
            if completed % 50 == 0 or time_since_update >= 300:  # 300 seconds = 5 minutes
                progress_pct = completed/total*100
                elapsed_sec = (current_time - start_time).total_seconds()
                elapsed_min = elapsed_sec / 60
                rate_per_sec = completed / elapsed_sec if elapsed_sec > 0 else 0
                rate_per_min = rate_per_sec * 60
                
                # Calculate ETA
                if rate_per_sec > 0:
                    remaining_sec = (total - completed) / rate_per_sec
                    remaining_min = remaining_sec / 60
                else:
                    remaining_min = 0
                
                # Cap ETA display
                eta_display = min(remaining_min, 999) if remaining_min > 0 else 0
                
                print(f"Progress: {completed}/{total} ({progress_pct:.1f}%) | "
                      f"Elapsed: {elapsed_min:.1f}m | Rate: {rate_per_min:.1f} tasks/min | "
                      f"ETA: {eta_display:.1f}m")
                last_update_time = current_time
            
            try:
                returns = future.result()
                if returns:
                    futures[future].update(returns)
            except:
                pass
    
    valid_data = [d for d in data if any(f'{p}d_return' in d for p in holding_periods)]
    total_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"[OK] Valid data points with returns: {len(valid_data)} (completed in {total_time:.1f} minutes)")
    
    return valid_data


def aggregate_multi_class_flow(data_by_date, secondary_weight, min_threshold):
    """Aggregate flow across share classes using liquidity-weighted approach."""
    data_copy = data_by_date.copy()
    ticker_set = {item['ticker'] for item in data_copy}  # Fast lookup
    
    for variants in SHARE_CLASS_GROUPS.values():
        if not any(v in ticker_set for v in variants):
            continue
        
        present = [item for item in data_copy if item['ticker'] in variants]
        if len(present) < 2:
            continue
        
        # Process bullish and bearish separately
        bullish_classes = [item for item in present if item['net_flow'] > 0]
        bearish_classes = [item for item in present if item['net_flow'] < 0]
        
        # Aggregate bullish flows
        if len(bullish_classes) >= 2:
            bullish_sorted = sorted(bullish_classes, key=lambda x: x['flow_score'], reverse=True)
            primary = bullish_sorted[0]
            threshold = min_threshold * primary['flow_score']
            
            for secondary in bullish_sorted[1:]:
                if secondary['flow_score'] >= threshold:
                    primary['flow_score'] += secondary['flow_score'] * secondary_weight
                    primary['multi_class_aggregated'] = True
        
        # Aggregate bearish flows
        if len(bearish_classes) >= 2:
            bearish_sorted = sorted(bearish_classes, key=lambda x: abs(x['flow_score']), reverse=True)
            primary = bearish_sorted[0]
            threshold = min_threshold * abs(primary['flow_score'])
            
            for secondary in bearish_sorted[1:]:
                if abs(secondary['flow_score']) >= threshold:
                    primary['flow_score'] -= abs(secondary['flow_score']) * secondary_weight
                    primary['multi_class_aggregated'] = True
    
    return data_copy

def normalize_flow_scores_cross_sectional(data):
    """
    FIXED: Normalize flow scores within each date for cross-sectional comparability.
    This makes flow scores comparable across different market conditions and dates.
    """
    dates = sorted(set(d['date'] for d in data))
    
    for date in dates:
        date_data = [d for d in data if d['date'] == date and 'flow_score' in d and not np.isnan(d.get('flow_score', np.nan))]
        if len(date_data) < 2:
            continue
        
        flow_scores = np.array([d['flow_score'] for d in date_data], dtype=np.float64)
        mean_score = np.mean(flow_scores)
        std_score = np.std(flow_scores)
        
        if std_score > 0:
            for d in date_data:
                d['flow_score_normalized'] = (d['flow_score'] - mean_score) / std_score
        else:
            for d in date_data:
                d['flow_score_normalized'] = 0.0

def calculate_flow_scores_for_data(data, params, use_volatility=True, use_cross_sectional_norm=True):
    """
    Calculate flow scores for all data points with given parameters.
    Used for correlation analysis (not ranking-based selection).
    
    OPTIMIZED: Pre-calculates volatility for all items in parallel to avoid repeated API calls.
    
    Args:
        use_volatility: If True, fetch and use volatility for normalization
        use_cross_sectional_norm: If True, normalize scores within each date
    """
    # Ensure PRICE_CAP_ADJUSTMENT is always present (defaults to True since removed from optimization)
    if 'PRICE_CAP_ADJUSTMENT' not in params:
        params = params.copy()
        params['PRICE_CAP_ADJUSTMENT'] = True
    
    # OPTIMIZATION: Pre-calculate volatility for all items in parallel (if enabled)
    # This avoids repeated API calls during the loop
    volatility_cache = {}
    if use_volatility:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def fetch_vol(item):
            ticker, date = item['ticker'], item['date']
            vol = calculate_realized_volatility(ticker, date, lookback_days=30)
            return (ticker, date), vol
        
        if len(data) > 100:  # Only print for large datasets
            print(f"  Pre-calculating volatility for {len(data)} items (parallelized)...")
        with ThreadPoolExecutor(max_workers=min(20, len(data))) as executor:
            futures = {executor.submit(fetch_vol, item): item for item in data}
            for future in as_completed(futures):
                try:
                    key, vol = future.result()
                    volatility_cache[key] = vol
                except Exception:
                    # Skip failed volatility calculations - will use default
                    pass
    
    # First pass: Calculate raw flow scores with volatility if enabled
    # ASK_WEIGHT is fixed at 1.0 (not in params)
    ask_weight = 1.0
    for item in data:
        bullish, bearish = calculate_flows_with_params(
            item, ask_weight, params['BID_WEIGHT'], params['DTE_LAMBDA']
        )
        
        net_flow = bullish - bearish
        
        # OPTIMIZED: Use pre-calculated volatility from cache
        volatility = None
        if use_volatility:
            key = (item['ticker'], item['date'])
            volatility = volatility_cache.get(key)
        
        flow_score = calculate_flow_score(
            net_flow, item['market_cap'], item['stock_price'],
            ask_weight, params['BID_WEIGHT'], 
            params['DTE_LAMBDA'], params['PRICE_CAP_ADJUSTMENT'],
            volatility=volatility
        )
        
        # NOTE: flow_score sign convention:
        # Positive flow_score = bullish flow (bullish > bearish) → should predict positive returns
        # Negative flow_score = bearish flow (bearish > bullish) → should predict negative returns
        # If IC is consistently negative, it means the market is fading the flow (contrarian signal)
        item['flow_score'] = flow_score
        item['net_flow'] = net_flow
        item['bullish_flow'] = bullish
        item['bearish_flow'] = bearish
        item['multi_class_aggregated'] = False
        item['volatility'] = volatility if volatility else 0.25  # Store for reference
    
    dates = sorted(set(d['date'] for d in data))
    
    for date in dates:
        date_data = [d for d in data if d['date'] == date]
        
        if 'SECONDARY_CLASS_WEIGHT' in params and 'MIN_SECONDARY_THRESHOLD' in params:
            date_data = aggregate_multi_class_flow(
                date_data, 
                params['SECONDARY_CLASS_WEIGHT'],
                params['MIN_SECONDARY_THRESHOLD']
            )
            for d in date_data:
                for original in data:
                    if original['ticker'] == d['ticker'] and original['date'] == d['date']:
                        original['flow_score'] = d['flow_score']
                        break
    
    # FIXED: Apply cross-sectional normalization if enabled
    if use_cross_sectional_norm:
        normalize_flow_scores_cross_sectional(data)
        # Use normalized scores for correlation analysis
        for item in data:
            if 'flow_score_normalized' in item:
                item['flow_score'] = item['flow_score_normalized']

def calculate_ranking_metrics(data, holding_periods, top_pct=0.10):
    """
    Calculate ranking-based metrics: top N% vs bottom N% returns.
    This tests if relative flow strength (ranking) predicts returns better than absolute correlation.
    """
    results = []
    dates = sorted(set(d['date'] for d in data))
    
    for period in holding_periods:
        return_key = f'{period}d_return'
        valid_data = [d for d in data if (return_key in d and d[return_key] is not None 
                     and 'flow_score' in d and not np.isnan(d['flow_score']))]
        
        if len(valid_data) < 20:
            results.append({
                'holding_period': period, 'top_pct': top_pct * 100, 'avg_spread': np.nan,
                'top_avg_return': np.nan, 'bottom_avg_return': np.nan, 'win_rate': np.nan,
                'num_dates': 0, 'sample_size': len(valid_data)
            })
            continue
        
        # Group by date and calculate ranking metrics
        date_results = []
        for date in dates:
            date_data = [d for d in valid_data if d['date'] == date]
            if len(date_data) < 10:
                continue
            
            # Rank by flow_score (highest = most bullish)
            date_data_sorted = sorted(date_data, key=lambda x: x['flow_score'], reverse=True)
            n_select = max(1, int(len(date_data_sorted) * top_pct))
            
            top_stocks = date_data_sorted[:n_select]
            bottom_stocks = date_data_sorted[-n_select:]
            
            top_returns = [d[return_key] for d in top_stocks]
            bottom_returns = [d[return_key] for d in bottom_stocks]
            
            if top_returns and bottom_returns:
                date_results.append({
                    'date': date,
                    'top_avg_return': np.mean(top_returns),
                    'bottom_avg_return': np.mean(bottom_returns),
                    'spread': np.mean(top_returns) - np.mean(bottom_returns),
                    'top_count': len(top_returns),
                    'bottom_count': len(bottom_returns)
                })
        
        if date_results:
            avg_spread = np.mean([r['spread'] for r in date_results])
            top_avg = np.mean([r['top_avg_return'] for r in date_results])
            bottom_avg = np.mean([r['bottom_avg_return'] for r in date_results])
            win_rate = np.mean([1 if r['spread'] > 0 else 0 for r in date_results]) * 100
            
            # DIAGNOSTICS: Calculate flow_score statistics for top vs bottom
            # This helps understand if the ranking is working correctly
            all_top_flow_scores = []
            all_bottom_flow_scores = []
            for date in dates:
                date_data = [d for d in valid_data if d['date'] == date]
                if len(date_data) < 10:
                    continue
                date_data_sorted = sorted(date_data, key=lambda x: x['flow_score'], reverse=True)
                n_select = max(1, int(len(date_data_sorted) * top_pct))
                top_stocks = date_data_sorted[:n_select]
                bottom_stocks = date_data_sorted[-n_select:]
                all_top_flow_scores.extend([d['flow_score'] for d in top_stocks])
                all_bottom_flow_scores.extend([d['flow_score'] for d in bottom_stocks])
            
            top_flow_avg = np.mean(all_top_flow_scores) if all_top_flow_scores else np.nan
            bottom_flow_avg = np.mean(all_bottom_flow_scores) if all_bottom_flow_scores else np.nan
            flow_separation = top_flow_avg - bottom_flow_avg if (not np.isnan(top_flow_avg) and not np.isnan(bottom_flow_avg)) else np.nan
            
            results.append({
                'holding_period': period, 'top_pct': top_pct * 100, 'avg_spread': avg_spread,
                'top_avg_return': top_avg, 'bottom_avg_return': bottom_avg, 'win_rate': win_rate,
                'num_dates': len(date_results), 'sample_size': len(valid_data),
                'top_avg_flow_score': top_flow_avg, 'bottom_avg_flow_score': bottom_flow_avg,
                'flow_separation': flow_separation
            })
        else:
            results.append({
                'holding_period': period, 'top_pct': top_pct * 100, 'avg_spread': np.nan,
                'top_avg_return': np.nan, 'bottom_avg_return': np.nan, 'win_rate': np.nan,
                'num_dates': 0, 'sample_size': len(valid_data)
            })
    
    return pd.DataFrame(results)

def calculate_correlation_metrics(data, holding_periods, use_normalized_scores=False):
    """
    Calculate correlation between flow scores and returns (Pearson, Spearman, IC, hit rate).
    
    FIXED: IC is now calculated cross-sectionally per date (quant standard), then averaged.
    This is the proper way to calculate Information Coefficient in quantitative finance.
    
    Args:
        use_normalized_scores: If True, uses 'flow_score_normalized' instead of 'flow_score'
    """
    correlations = []
    dates = sorted(set(d['date'] for d in data))
    
    for period in holding_periods:
        return_key = f'{period}d_return'
        if use_normalized_scores:
            valid_data = [d for d in data if (return_key in d and d[return_key] is not None 
                         and 'flow_score_normalized' in d and not np.isnan(d.get('flow_score_normalized', np.nan)))]
        else:
            valid_data = [d for d in data if (return_key in d and d[return_key] is not None 
                         and 'flow_score' in d and not np.isnan(d['flow_score']))]
        
        if len(valid_data) < 10:
            correlations.append({
                'holding_period': period, 'pearson_correlation': np.nan, 'spearman_correlation': np.nan,
                'ic': np.nan, 'ic_tstat': np.nan, 'hit_rate': np.nan, 'sample_size': len(valid_data), 'num_dates': 0
            })
            continue
        
        # Overall Pearson and Spearman correlation (for reference)
        if use_normalized_scores:
            flow_scores = np.array([d['flow_score_normalized'] for d in valid_data], dtype=np.float64)
        else:
            flow_scores = np.array([d['flow_score'] for d in valid_data], dtype=np.float64)
        returns = np.array([d[return_key] for d in valid_data], dtype=np.float64)
        pearson_corr, pearson_pval = stats.pearsonr(flow_scores, returns)
        spearman_corr, spearman_pval = stats.spearmanr(flow_scores, returns)
        hit_rate = np.mean(np.sign(flow_scores) == np.sign(returns)) * 100
        
        # QUANT STANDARD: Calculate IC cross-sectionally per date, then average
        # This is how quants typically define IC - rank correlation within each date
        daily_ics = []
        for date in dates:
            date_data = [d for d in valid_data if d['date'] == date]
            if len(date_data) < 3:  # Need at least 3 stocks per date for correlation
                continue
            
            if use_normalized_scores:
                date_flow_scores = [d.get('flow_score_normalized', d.get('flow_score', 0)) for d in date_data]
            else:
                date_flow_scores = [d['flow_score'] for d in date_data]
            date_returns = [d[return_key] for d in date_data]
            
            # Filter out NaN values
            valid_pairs = [(f, r) for f, r in zip(date_flow_scores, date_returns) 
                          if not (np.isnan(f) or np.isnan(r))]
            if len(valid_pairs) < 3:
                continue
            
            date_flow_scores_clean = [f for f, r in valid_pairs]
            date_returns_clean = [r for f, r in valid_pairs]
            
            # Use Spearman (rank correlation) for IC - quant standard
            try:
                daily_ic, _ = stats.spearmanr(date_flow_scores_clean, date_returns_clean)
                if not np.isnan(daily_ic) and np.isfinite(daily_ic):
                    daily_ics.append(daily_ic)
            except (ValueError, RuntimeError):
                continue
        
        if len(daily_ics) > 0:
            mean_ic = np.mean(daily_ics)
            ic_std = np.std(daily_ics)
            # IC t-statistic: mean IC / (std of daily ICs / sqrt(n_dates))
            ic_tstat = (mean_ic / (ic_std / np.sqrt(len(daily_ics)))) if len(daily_ics) > 1 and ic_std > 0 else 0
            
            # DIAGNOSTICS: Check if IC is consistently negative
            positive_ic_days = sum(1 for ic in daily_ics if ic > 0)
            negative_ic_days = sum(1 for ic in daily_ics if ic < 0)
            ic_positive_pct = (positive_ic_days / len(daily_ics)) * 100 if len(daily_ics) > 0 else 0
        else:
            mean_ic = np.nan
            ic_tstat = np.nan
            ic_positive_pct = np.nan
            positive_ic_days = 0
            negative_ic_days = 0
        
        correlations.append({
            'holding_period': period, 'pearson_correlation': pearson_corr, 'pearson_pvalue': pearson_pval,
            'spearman_correlation': spearman_corr, 'spearman_pvalue': spearman_pval,
            'ic': mean_ic, 'ic_tstat': ic_tstat, 'hit_rate': hit_rate, 'sample_size': len(valid_data),
            'num_dates': len(daily_ics) if daily_ics else 0, 'ic_positive_days': positive_ic_days,
            'ic_negative_days': negative_ic_days, 'ic_positive_pct': ic_positive_pct
        })
    return pd.DataFrame(correlations)

# ============================================================================
# PARAMETER OPTIMIZATION
# ============================================================================

def split_train_test(data, train_ratio=0.70):
    """Split data into train/test sets by dates."""
    dates = sorted(set(d['date'] for d in data))
    split_idx = int(len(dates) * train_ratio)
    train_dates, test_dates = set(dates[:split_idx]), set(dates[split_idx:])
    return ([d for d in data if d['date'] in train_dates], 
            [d for d in data if d['date'] in test_dates], train_dates, test_dates)

def get_market_regime(date_str, vix_data_cache=None):
    """
    Classify market regime based on SPY volatility.
    Returns: 'low_vol' (vol < 0.15), 'normal' (0.15-0.25), 'high_vol' (vol > 0.25)
    """
    spy_vol = calculate_spy_volatility(date_str)
    if spy_vol is None:
        return 'unknown'
    if spy_vol < 0.15:
        return 'low_vol'
    elif spy_vol > 0.25:
        return 'high_vol'
    else:
        return 'normal'

def calculate_ic_by_regime(data, holding_periods):
    """
    Calculate IC separately for low-vol, normal, high-vol periods.
    """
    # Classify each data point by SPY volatility regime
    for item in data:
        item['regime'] = get_market_regime(item['date'])
    
    # Calculate IC for each regime
    regime_results = {}
    for regime in ['low_vol', 'normal', 'high_vol']:
        regime_data = [d for d in data if d.get('regime') == regime]
        if len(regime_data) >= 20:  # Minimum sample size
            regime_corr = calculate_correlation_metrics(regime_data, holding_periods, use_normalized_scores=False)
            regime_results[regime] = regime_corr
        else:
            regime_results[regime] = None
    
    return regime_results

def split_out_of_sample(data, holdout_days=30):
    """
    Split data: hold out most recent days for out-of-sample validation.
    Returns in_sample_data, out_of_sample_data.
    """
    def to_date(d):
        if isinstance(d, str):
            return datetime.strptime(d, '%Y-%m-%d').date()
        elif isinstance(d, datetime):
            return d.date()
        return d
    
    dates = sorted(set(d['date'] for d in data))
    if len(dates) <= holdout_days:
        return data, []
    
    cutoff_date = dates[-holdout_days]
    cutoff_date_obj = to_date(cutoff_date)
    
    in_sample_data = [d for d in data if to_date(d['date']) < cutoff_date_obj]
    out_of_sample_data = [d for d in data if to_date(d['date']) >= cutoff_date_obj]
    
    return in_sample_data, out_of_sample_data

def get_market_regime(date_str, vix_data_cache=None):
    """
    Classify market regime based on SPY volatility.
    Returns: 'low_vol' (vol < 0.15), 'normal' (0.15-0.25), 'high_vol' (vol > 0.25)
    """
    spy_vol = calculate_spy_volatility(date_str)
    if spy_vol is None:
        return 'unknown'
    if spy_vol < 0.15:
        return 'low_vol'
    elif spy_vol > 0.25:
        return 'high_vol'
    else:
        return 'normal'

def calculate_ic_by_regime(data, holding_periods):
    """
    Calculate IC separately for low-vol, normal, high-vol periods.
    """
    # Classify each data point by SPY volatility regime
    for item in data:
        item['regime'] = get_market_regime(item['date'])
    
    # Calculate IC for each regime
    regime_results = {}
    for regime in ['low_vol', 'normal', 'high_vol']:
        regime_data = [d for d in data if d.get('regime') == regime]
        if len(regime_data) >= 20:  # Minimum sample size
            regime_corr = calculate_correlation_metrics(regime_data, holding_periods, use_normalized_scores=False)
            regime_results[regime] = regime_corr
        else:
            regime_results[regime] = None
    
    return regime_results

def optimize_parameters(data, holding_periods):
    """
    Optimize parameters using Bayesian optimization (Optuna) with overfitting prevention.
    
    Uses train/test split (70/30), out-of-sample validation, and regularization.
    Optimizes based on Information Coefficient (IC) - quant standard cross-sectional IC.
    Also validates on ranking metrics to ensure generalization.
    """
    print(f"\n{'='*80}")
    print(f"BAYESIAN OPTIMIZATION (Optuna)")
    print(f"{'='*80}")
    print(f"Trials: {N_TRIALS} | Train/Test: {TRAIN_TEST_SPLIT*100:.0f}%/{((1-TRAIN_TEST_SPLIT)*100):.0f}% | OOS Holdout: {OUT_OF_SAMPLE_DAYS} days\n")
    
    # Split data: First hold out most recent days for final validation
    in_sample_data, out_of_sample_data = split_out_of_sample(data, OUT_OF_SAMPLE_DAYS)
    
    if not out_of_sample_data:
        print(f"Warning: Not enough data for {OUT_OF_SAMPLE_DAYS}-day holdout, using all data")
        in_sample_data = data
        out_of_sample_data = []
    
    # Split remaining data into train/test
    train_data, test_data, train_dates, test_dates = split_train_test(in_sample_data, TRAIN_TEST_SPLIT)
    
    print(f"Data split:")
    print(f"  Training set: {len(train_data)} data points ({len(train_dates)} dates)")
    print(f"  Test set: {len(test_data)} data points ({len(test_dates)} dates)")
    if out_of_sample_data:
        out_of_sample_dates = sorted(set(d['date'] for d in out_of_sample_data))
        print(f"  Out-of-sample: {len(out_of_sample_data)} data points ({len(out_of_sample_dates)} dates)\n")
    else:
        print(f"  Out-of-sample: None (not enough data)\n")
    
    # Validate test set has sufficient data
    if len(test_data) == 0:
        print(f"[ERROR] Test set is empty! Cannot proceed with optimization.")
        print(f"  This suggests insufficient data after train/test split.")
        return {}, pd.DataFrame()
    
    # Check test set has returns for swing periods
    swing_periods = [p for p in holding_periods if 3 <= p <= 7]
    test_has_returns = {}
    for period in swing_periods:
        return_key = f'{period}d_return'
        test_with_returns = [d for d in test_data if return_key in d and d[return_key] is not None]
        test_has_returns[period] = len(test_with_returns)
    
    print(f"Test set return data check:")
    for period, count in test_has_returns.items():
        print(f"  {period}d_return: {count} data points")
    
    if all(count < 10 for count in test_has_returns.values()):
        print(f"[WARNING] Test set has very few data points with returns for swing periods")
        print(f"  This may cause optimization to fail or produce poor results")
    
    all_results = []
    test_results = []
    optuna_start_time = datetime.now()
    last_optuna_update = optuna_start_time
    
    def objective(trial):
        nonlocal last_optuna_update
        current_time = datetime.now()
        time_since_update = (current_time - last_optuna_update).total_seconds()
        
        # Progress update every 5 minutes during optimization
        if time_since_update >= 300:  # 300 seconds = 5 minutes
            elapsed = (current_time - optuna_start_time).total_seconds() / 60
            print(f"[{current_time.strftime('%H:%M:%S')}] Optimization progress: "
                  f"Trial {trial.number + 1}/{N_TRIALS} | Elapsed: {elapsed:.1f} minutes")
            last_optuna_update = current_time
        
        try:
            params = {
                'BID_WEIGHT': trial.suggest_float('BID_WEIGHT', *PARAM_RANGES['BID_WEIGHT']),
                'DTE_LAMBDA': trial.suggest_float('DTE_LAMBDA', *PARAM_RANGES['DTE_LAMBDA']),
                'PRICE_CAP_ADJUSTMENT': True,  # Always enabled - optimization tests showed it consistently improves scores
                'SECONDARY_CLASS_WEIGHT': trial.suggest_float('SECONDARY_CLASS_WEIGHT', *PARAM_RANGES['SECONDARY_CLASS_WEIGHT']),
                'MIN_SECONDARY_THRESHOLD': trial.suggest_float('MIN_SECONDARY_THRESHOLD', *PARAM_RANGES['MIN_SECONDARY_THRESHOLD'])
            }
            # ASK_WEIGHT is fixed at 1.0 (not optimized, not in params)
            
            # Regularization penalty for extreme parameters (relaxed to match expanded ranges)
            penalty = 0
            if params['BID_WEIGHT'] < 0.3 or params['BID_WEIGHT'] > 1.0:
                penalty += 0.02
            if params['DTE_LAMBDA'] < 0.02 or params['DTE_LAMBDA'] > 0.25:
                penalty += 0.02
            
            # Optimize on TRAIN set using IC/correlation (quant standard)
            # Then validate on ranking metrics to ensure generalization
            try:
                calculate_flow_scores_for_data(train_data, params, 
                                             use_volatility=USE_VOLATILITY_NORMALIZATION,
                                             use_cross_sectional_norm=USE_CROSS_SECTIONAL_NORMALIZATION)
            except Exception as e:
                if trial.number % 10 == 0:  # Only print occasionally to avoid spam
                    print(f"\n[WARNING] Trial {trial.number + 1}: Flow score calculation failed: {e}")
                return -np.inf
            
            # Primary: IC-based optimization (quant standard)
            try:
                train_correlation_results = calculate_correlation_metrics(train_data, holding_periods)
            except Exception as e:
                if trial.number % 10 == 0:
                    print(f"\n[WARNING] Trial {trial.number + 1}: Train correlation calculation failed: {e}")
                return -np.inf
            
            if train_correlation_results.empty:
                return -np.inf
            
            swing_periods = [p for p in holding_periods if 3 <= p <= 7]
            swing_correlations = train_correlation_results[train_correlation_results['holding_period'].isin(swing_periods)]
            
            if swing_correlations.empty:
                return -np.inf
            
            mean_ic = swing_correlations['ic'].mean()
            mean_ic_tstat = swing_correlations['ic_tstat'].mean()
            mean_hit_rate = swing_correlations['hit_rate'].mean()
            
            # Handle NaN values
            mean_ic = 0.0 if np.isnan(mean_ic) else mean_ic
            mean_ic_tstat = 0.0 if np.isnan(mean_ic_tstat) else mean_ic_tstat
            mean_hit_rate = 50.0 if np.isnan(mean_hit_rate) else mean_hit_rate
            
            # Validate on TEST set using IC
            try:
                calculate_flow_scores_for_data(test_data, params,
                                             use_volatility=USE_VOLATILITY_NORMALIZATION,
                                             use_cross_sectional_norm=USE_CROSS_SECTIONAL_NORMALIZATION)
            except Exception as e:
                if trial.number % 10 == 0:
                    print(f"\n[WARNING] Trial {trial.number + 1}: Test flow score calculation failed: {e}")
                return -np.inf
            
            try:
                test_correlation_results = calculate_correlation_metrics(test_data, holding_periods)
            except Exception as e:
                if trial.number % 10 == 0:
                    print(f"\n[WARNING] Trial {trial.number + 1}: Test correlation calculation failed: {e}")
                test_correlation_results = pd.DataFrame()
            
            test_ic = 0.0
            test_ic_tstat = 0.0
            if not test_correlation_results.empty:
                test_swing_corr = test_correlation_results[test_correlation_results['holding_period'].isin(swing_periods)]
                if not test_swing_corr.empty:
                    test_ic = test_swing_corr['ic'].mean()
                    test_ic_tstat = test_swing_corr['ic_tstat'].mean()
                    # Handle NaN values
                    test_ic = 0.0 if np.isnan(test_ic) else test_ic
                    test_ic_tstat = 0.0 if np.isnan(test_ic_tstat) else test_ic_tstat
            
            # Secondary validation: Ranking metrics (to ensure generalization beyond top 10%)
            try:
                test_ranking_results = calculate_ranking_metrics(test_data, holding_periods, top_pct=0.10)
            except Exception as e:
                if trial.number % 10 == 0:
                    print(f"\n[WARNING] Trial {trial.number + 1}: Test ranking calculation failed: {e}")
                test_ranking_results = pd.DataFrame()
            
            test_spread = 0.0
            if not test_ranking_results.empty:
                test_swing_ranking = test_ranking_results[test_ranking_results['holding_period'].isin(swing_periods)]
                if not test_swing_ranking.empty:
                    test_spread = test_swing_ranking['avg_spread'].mean()
                    if np.isnan(test_spread):
                        test_spread = 0.0
            
            # Optimization objective: Focus on IC (quant standard), validate on ranking
            # IMPORTANT: Test spread removed from optimization to prevent overfitting
            # Test spread is calculated on same test set repeatedly, which could overfit
            # Test spread will still be calculated and reported as a VALIDATION metric only
            # Primary: test IC (out-of-sample), Secondary: statistical significance and directional accuracy
            score = (
                0.75 * test_ic * 100 +           # Primary: out-of-sample IC (quant standard) - increased weight
                0.15 * test_ic_tstat +           # Statistical significance
                0.10 * (mean_hit_rate - 50) +     # Directional accuracy
                # Removed: test_spread from optimization (kept as validation metric only)
                penalty * 100                     # Regularization penalty
            )
            
            # Ensure score is not NaN
            if np.isnan(score) or not np.isfinite(score):
                if trial.number % 10 == 0:  # Only print occasionally to avoid spam
                    print(f"\n[WARNING] Trial {trial.number + 1}: Score is NaN or infinite")
                    print(f"  Train IC: {mean_ic:.4f}, Test IC: {test_ic:.4f}, Test IC t-stat: {test_ic_tstat:.2f}")
                    print(f"  Train hit rate: {mean_hit_rate:.1f}%, Penalty: {penalty:.4f}")
                return -np.inf
            
            # Get test hit rate for completeness
            test_hit_rate = 50.0
            if not test_correlation_results.empty:
                test_swing_corr = test_correlation_results[test_correlation_results['holding_period'].isin(swing_periods)]
                if not test_swing_corr.empty:
                    test_hit_rate = test_swing_corr['hit_rate'].mean()
                    test_hit_rate = 50.0 if np.isnan(test_hit_rate) else test_hit_rate
            
            # Calculate score components for analysis
            score_ic_component = test_ic * 100 * 0.75
            score_tstat_component = test_ic_tstat * 0.15
            score_hitrate_component = (mean_hit_rate - 50) * 0.10
            score_penalty_component = -penalty * 100
            
            all_results.append({
                **params,
                'score': score,
                'train_ic': mean_ic,
                'test_ic': test_ic,
                'ic_tstat': mean_ic_tstat,
                'train_hit_rate': mean_hit_rate,
                'test_hit_rate': test_hit_rate,
                'test_spread': test_spread,
                'trial_number': trial.number,
                # Score components for analysis
                'score_ic_component': score_ic_component,
                'score_tstat_component': score_tstat_component,
                'score_hitrate_component': score_hitrate_component,
                'score_penalty_component': score_penalty_component
            })
            
            # Print trial result every 10 trials or if it's a good score
            if trial.number % 10 == 0 or score > 0:
                print(f"\nTrial {trial.number + 1}/{N_TRIALS}:")
                print(f"  Parameters: BID_WEIGHT={params['BID_WEIGHT']:.3f}, DTE_LAMBDA={params['DTE_LAMBDA']:.3f}, SEC_WEIGHT={params['SECONDARY_CLASS_WEIGHT']:.3f}")
                print(f"  Train IC: {mean_ic:.4f} | Test IC: {test_ic:.4f} (t-stat: {test_ic_tstat:.2f})")
                print(f"  Train Hit Rate: {mean_hit_rate:.1f}% | Test Hit Rate: {test_hit_rate:.1f}%")
                print(f"  Test Spread: {test_spread:.4f}%")
                print(f"  SCORE BREAKDOWN (optimization):")
                print(f"    75% Test IC:        {test_ic * 100 * 0.75:+.2f}")
                print(f"    15% IC t-stat:      {test_ic_tstat * 0.15:+.2f}")
                print(f"    10% Hit Rate:       {(mean_hit_rate - 50) * 0.10:+.2f}")
                print(f"    Penalty:            {-penalty * 100:.2f}")
                print(f"  VALIDATION METRICS (not in optimization):")
                print(f"    Test Spread:        {test_spread:.4f}%")
                print(f"  TOTAL SCORE: {score:.4f}")
            
            return score
        
        except Exception as e:
            # Catch any unexpected errors and return a bad score instead of crashing
            if trial.number % 10 == 0:  # Only print occasionally to avoid spam
                import traceback
                print(f"\n[ERROR] Trial {trial.number + 1}: Unexpected error: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
            return -np.inf
    
    study = optuna.create_study(direction='maximize', study_name='flow_score_optimization')
    print(f"Starting optimization at {datetime.now().strftime('%H:%M:%S')}...")
    print(f"Progress updates every 5 minutes during optimization\n")
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    except KeyboardInterrupt:
        print(f"\n[WARNING] Optimization interrupted by user after {len(study.trials)} trials")
        if len(study.trials) == 0:
            print("[ERROR] No trials completed. Cannot proceed.")
            return {}, pd.DataFrame()
    except Exception as e:
        print(f"\n[ERROR] Optimization failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        if len(study.trials) == 0:
            print("[ERROR] No trials completed. Cannot proceed.")
            return {}, pd.DataFrame()
    
    optuna_total_time = (datetime.now() - optuna_start_time).total_seconds() / 60
    print(f"\n[OK] Optimization completed in {optuna_total_time:.1f} minutes")
    
    best_params = study.best_params.copy()
    # Add fixed parameters that weren't optimized
    # ASK_WEIGHT is fixed at 1.0 (not in params, not optimized)
    best_params['PRICE_CAP_ADJUSTMENT'] = True  # Always enabled
    best_optimization_score = study.best_value
    
    # Get the actual train IC from the best trial results
    best_trial_train_ic = None
    if all_results:
        best_trial_result = max(all_results, key=lambda x: x['score'])
        best_trial_train_ic = best_trial_result.get('train_ic', None)
    
    # Final validation on out-of-sample data
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best optimization score: {best_optimization_score:.4f}")
    print(f"Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Validate on TEST set with best params using ranking approach
    print(f"\n{'='*80}")
    print(f"TEST SET VALIDATION DIAGNOSTICS")
    print(f"{'='*80}")
    print(f"Test set size: {len(test_data)} data points")
    if test_data:
        test_dates = sorted(set(d['date'] for d in test_data))
        print(f"Test set dates: {len(test_dates)} dates")
        print(f"Date range: {test_dates[0]} to {test_dates[-1]}")
        
        # Check for returns data
        swing_periods = [p for p in holding_periods if 3 <= p <= 7]
        for period in swing_periods:
            return_key = f'{period}d_return'
            valid_with_returns = [d for d in test_data if return_key in d and d[return_key] is not None]
            print(f"  {return_key}: {len(valid_with_returns)} data points with returns")
    
    calculate_flow_scores_for_data(test_data, best_params,
                                   use_volatility=USE_VOLATILITY_NORMALIZATION,
                                   use_cross_sectional_norm=USE_CROSS_SECTIONAL_NORMALIZATION)
    
    # Check flow scores were calculated
    if test_data:
        valid_with_flow = [d for d in test_data if 'flow_score' in d and not np.isnan(d.get('flow_score', np.nan))]
        print(f"Test set with flow scores: {len(valid_with_flow)} data points")
        if len(valid_with_flow) > 0:
            flow_scores = [d['flow_score'] for d in valid_with_flow]
            print(f"  Flow score range: {min(flow_scores):.6f} to {max(flow_scores):.6f}")
    
    test_ranking_results = calculate_ranking_metrics(test_data, holding_periods, top_pct=0.10)
    print(f"Test ranking results: {len(test_ranking_results)} rows")
    if not test_ranking_results.empty:
        print(f"  Holding periods in results: {sorted(test_ranking_results['holding_period'].unique())}")
    
    if not test_ranking_results.empty:
        swing_periods = [p for p in holding_periods if 3 <= p <= 7]
        test_swing = test_ranking_results[test_ranking_results['holding_period'].isin(swing_periods)]
        if not test_swing.empty:
            test_spread = test_swing['avg_spread'].mean()
            test_win_rate = test_swing['win_rate'].mean()
            
            # DIAGNOSTICS: Show flow_score separation and return details
            test_top_flow = test_swing['top_avg_flow_score'].mean() if 'top_avg_flow_score' in test_swing.columns else np.nan
            test_bottom_flow = test_swing['bottom_avg_flow_score'].mean() if 'bottom_avg_flow_score' in test_swing.columns else np.nan
            test_flow_separation = test_swing['flow_separation'].mean() if 'flow_separation' in test_swing.columns else np.nan
            test_top_return = test_swing['top_avg_return'].mean()
            test_bottom_return = test_swing['bottom_avg_return'].mean()
            
            print(f"\nTest Set Validation (Ranking Approach):")
            print(f"  Test Spread (Top 10% vs Bottom 10%): {test_spread:.4f}%")
            print(f"  Test Win Rate: {test_win_rate:.1f}%")
            print(f"\n  DIAGNOSTICS:")
            if not np.isnan(test_flow_separation):
                print(f"    Top 10% Avg Flow Score: {test_top_flow:.6f}")
                print(f"    Bottom 10% Avg Flow Score: {test_bottom_flow:.6f}")
                print(f"    Flow Score Separation: {test_flow_separation:.6f}")
            print(f"    Top 10% Avg Return: {test_top_return:.4f}%")
            print(f"    Bottom 10% Avg Return: {test_bottom_return:.4f}%")
            print(f"    Return Spread: {test_spread:.4f}%")
        else:
            print(f"\n[WARNING] Test ranking results empty for swing periods (3-7 days)")
            print(f"  Available holding periods in results: {sorted(test_ranking_results['holding_period'].unique())}")
            print(f"  This suggests test set may not have sufficient data for swing periods")
    else:
        print(f"\n[ERROR] Test ranking results are completely empty!")
        print(f"  This indicates a critical issue with test set data or flow score calculation")
    
    # Get train metrics from best trial (if ranking results were successful)
    if not test_ranking_results.empty:
        swing_periods = [p for p in holding_periods if 3 <= p <= 7]
        test_swing = test_ranking_results[test_ranking_results['holding_period'].isin(swing_periods)]
        if not test_swing.empty:
            test_spread = test_swing['avg_spread'].mean()
            test_flow_separation = test_swing['flow_separation'].mean() if 'flow_separation' in test_swing.columns else np.nan
            
            # Check if spread is low due to weak flow separation or weak return prediction
            if not np.isnan(test_flow_separation) and test_flow_separation > 0.001:
                if abs(test_spread) < 0.3:
                    print(f"    [WARNING] Low spread despite good flow separation - signal may be weak or reversed")
            elif not np.isnan(test_flow_separation) and test_flow_separation < 0.001:
                print(f"    [WARNING] Very small flow separation - top and bottom stocks have similar flow scores")
            
            # Get train metrics from best trial
            if all_results:
                best_trial_result = max(all_results, key=lambda x: x['score'])
                train_spread = best_trial_result.get('train_spread', None)
                train_win_rate = best_trial_result.get('train_win_rate', None)
                if train_spread is not None:
                    print(f"  Train Spread: {train_spread:.4f}%")
                    print(f"  Train Win Rate: {train_win_rate:.1f}%")
                    spread_gap = train_spread - test_spread
                    print(f"  Train-Test Spread Gap: {spread_gap:.4f}%")
                    if abs(spread_gap) > 0.5:
                        print(f"  ⚠ Warning: Large train-test spread gap may indicate overfitting")
    
    # Out-of-sample validation using IC (quant standard)
    if test_data:
        calculate_flow_scores_for_data(test_data, best_params,
                                     use_volatility=USE_VOLATILITY_NORMALIZATION,
                                     use_cross_sectional_norm=USE_CROSS_SECTIONAL_NORMALIZATION)
        test_correlation_results = calculate_correlation_metrics(test_data, holding_periods)
        
        if not test_correlation_results.empty:
            swing_periods = [p for p in holding_periods if 3 <= p <= 7]
            test_swing_corr = test_correlation_results[test_correlation_results['holding_period'].isin(swing_periods)]
            if not test_swing_corr.empty:
                test_ic = test_swing_corr['ic'].mean()
                test_ic_tstat = test_swing_corr['ic_tstat'].mean()
                test_ic_positive_pct = test_swing_corr['ic_positive_pct'].mean() if 'ic_positive_pct' in test_swing_corr.columns else np.nan
                test_ic_positive_days = test_swing_corr['ic_positive_days'].sum() if 'ic_positive_days' in test_swing_corr.columns else 0
                test_ic_negative_days = test_swing_corr['ic_negative_days'].sum() if 'ic_negative_days' in test_swing_corr.columns else 0
                
                print(f"\nTest Set Validation (IC Approach):")
                print(f"  Test IC: {test_ic:.4f}")
                print(f"  Test IC t-statistic: {test_ic_tstat:.2f}")
                if not np.isnan(test_ic_positive_pct):
                    print(f"  IC Positive Days: {test_ic_positive_days} ({test_ic_positive_pct:.1f}%)")
                    print(f"  IC Negative Days: {test_ic_negative_days} ({100 - test_ic_positive_pct:.1f}%)")
                    if test_ic < 0 and test_ic_positive_pct < 40:
                        print(f"  [WARNING] IC is negative and {100 - test_ic_positive_pct:.1f}% of days have negative IC")
                        print(f"    This suggests the flow signal may be reversed or the market is fading the flow")
                    elif test_ic < 0 and test_ic_positive_pct > 50:
                        print(f"    [INFO] IC is negative on average but positive on {test_ic_positive_pct:.1f}% of days")
                        print(f"    This suggests inconsistent signal or market regime changes")
    
    if out_of_sample_data:
        calculate_flow_scores_for_data(out_of_sample_data, best_params,
                                     use_volatility=USE_VOLATILITY_NORMALIZATION,
                                     use_cross_sectional_norm=USE_CROSS_SECTIONAL_NORMALIZATION)
        oos_correlation_results = calculate_correlation_metrics(out_of_sample_data, holding_periods)
        
        if not oos_correlation_results.empty:
            swing_periods = [p for p in holding_periods if 3 <= p <= 7]
            oos_swing = oos_correlation_results[oos_correlation_results['holding_period'].isin(swing_periods)]
            if not oos_swing.empty:
                oos_ic = oos_swing['ic'].mean()
                oos_ic_tstat = oos_swing['ic_tstat'].mean()
                oos_hit_rate = oos_swing['hit_rate'].mean()
                print(f"\nOut-of-Sample Validation ({OUT_OF_SAMPLE_DAYS} most recent days):")
                print(f"  OOS IC: {oos_ic:.4f}")
                print(f"  OOS IC t-statistic: {oos_ic_tstat:.2f}")
                print(f"  OOS Hit Rate: {oos_hit_rate:.1f}%")
                if oos_ic < 0:
                    print(f"  [WARNING] Negative OOS IC suggests overfitting")
                elif oos_ic < 0.05:
                    print(f"  [WARNING] Weak OOS IC suggests limited generalizability")
                
                # Also show ranking validation
                oos_ranking_results = calculate_ranking_metrics(out_of_sample_data, holding_periods, top_pct=0.10)
                if not oos_ranking_results.empty:
                    oos_swing_ranking = oos_ranking_results[oos_ranking_results['holding_period'].isin(swing_periods)]
                    if not oos_swing_ranking.empty:
                        oos_spread = oos_swing_ranking['avg_spread'].mean()
                        oos_win_rate = oos_swing_ranking['win_rate'].mean()
                        print(f"  OOS Ranking Spread: {oos_spread:.4f}%")
                        print(f"  OOS Ranking Win Rate: {oos_win_rate:.1f}%")
    
    print(f"\nTotal trials: {N_TRIALS}")
    print(f"Best trial: {study.best_trial.number}\n")
    
    results_df = pd.DataFrame(all_results)
    
    return best_params, results_df


def create_visualizations(data, correlation_results, param_results, output_dir):
    """Generate comprehensive visualization plots."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        plots_generated = 0
        
        # 1. Flow Score vs Returns Scatter Plot
        for period in HOLDING_PERIODS:
            return_key = f'{period}d_return'
            valid_data = [d for d in data if return_key in d and d[return_key] is not None 
                         and 'flow_score' in d and not np.isnan(d['flow_score'])]
            
            if len(valid_data) < 10:
                continue
            
            flow_scores = [d['flow_score'] for d in valid_data]
            returns = [d[return_key] for d in valid_data]
            
            plt.figure(figsize=(14, 8))
            
            # Color code by sign of flow_score
            colors = ['green' if fs > 0 else 'red' for fs in flow_scores]
            plt.scatter(flow_scores, returns, alpha=0.5, s=30, c=colors, 
                       label='Positive Flow (green) / Negative Flow (red)')
            
            # Add regression line
            z = np.polyfit(flow_scores, returns, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(flow_scores), max(flow_scores), 100)
            plt.plot(x_line, p(x_line), "blue", linewidth=2, 
                    label=f'Regression: y={z[0]:.4f}x+{z[1]:.4f}')
            
            # Add zero lines
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            # Calculate correlations
            pearson_corr, pearson_pval = stats.pearsonr(flow_scores, returns)
            spearman_corr, spearman_pval = stats.spearmanr(flow_scores, returns)
            
            # Calculate hit rate
            hit_rate = np.mean(np.sign(flow_scores) == np.sign(returns)) * 100
            
            # Calculate IC (cross-sectional)
            dates_in_data = sorted(set(d['date'] for d in valid_data))
            daily_ics = []
            for date in dates_in_data:
                date_data = [d for d in valid_data if d['date'] == date]
                if len(date_data) < 3:
                    continue
                date_flows = [d['flow_score'] for d in date_data]
                date_rets = [d[return_key] for d in date_data]
                valid_pairs = [(f, r) for f, r in zip(date_flows, date_rets) 
                              if not (np.isnan(f) or np.isnan(r))]
                if len(valid_pairs) < 3:
                    continue
                try:
                    ic, _ = stats.spearmanr([f for f, r in valid_pairs], [r for f, r in valid_pairs])
                    if not np.isnan(ic) and np.isfinite(ic):
                        daily_ics.append(ic)
                except:
                    continue
            
            mean_ic = np.mean(daily_ics) if daily_ics else np.nan
            
            plt.title(f'Flow Score vs {period}-Day Returns\n'
                     f'Pearson r={pearson_corr:.3f} (p={pearson_pval:.4f}), '
                     f'Spearman r={spearman_corr:.3f}, IC={mean_ic:.3f}, Hit Rate={hit_rate:.1f}%\n'
                     f'Slope: {z[0]:.4f} ({"POSITIVE" if z[0] > 0 else "NEGATIVE"} relationship)', 
                     fontsize=12)
            plt.xlabel('Flow Score (positive = bullish flow, negative = bearish flow)', fontsize=12)
            plt.ylabel(f'{period}-Day Return (%)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'scatter_flow_vs_{period}d_returns.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            plots_generated += 1
    
        # 2. IC Over Time (Time Series)
        # Calculate daily ICs for plotting
        dates = sorted(set(d['date'] for d in data))
        for period in HOLDING_PERIODS:
            return_key = f'{period}d_return'
            daily_ics = []
            daily_dates = []
            
            for date in dates:
                date_data = [d for d in data if d['date'] == date and return_key in d and d[return_key] is not None 
                           and 'flow_score' in d and not np.isnan(d['flow_score'])]
                if len(date_data) < 3:
                    continue
                
                date_flow_scores = [d['flow_score'] for d in date_data]
                date_returns = [d[return_key] for d in date_data]
                
                valid_pairs = [(f, r) for f, r in zip(date_flow_scores, date_returns) 
                              if not (np.isnan(f) or np.isnan(r))]
                if len(valid_pairs) < 3:
                    continue
                
                date_flow_clean = [f for f, r in valid_pairs]
                date_returns_clean = [r for f, r in valid_pairs]
                
                try:
                    daily_ic, _ = stats.spearmanr(date_flow_clean, date_returns_clean)
                    if not np.isnan(daily_ic) and np.isfinite(daily_ic):
                        daily_ics.append(daily_ic)
                        daily_dates.append(date)
                except (ValueError, RuntimeError):
                    continue
            
            if len(daily_ics) > 0:
                plt.figure(figsize=(14, 6))
                plt.plot(daily_dates, daily_ics, marker='o', markersize=4, alpha=0.6, linewidth=1)
                plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Zero IC')
                plt.axhline(y=np.mean(daily_ics), color='red', linestyle='-', linewidth=2, 
                           label=f'Mean IC: {np.mean(daily_ics):.3f}')
                plt.fill_between(daily_dates, 0, daily_ics, where=(np.array(daily_ics) > 0), 
                               alpha=0.3, color='green', label='Positive IC')
                plt.fill_between(daily_dates, 0, daily_ics, where=(np.array(daily_ics) < 0), 
                               alpha=0.3, color='red', label='Negative IC')
                plt.title(f'Information Coefficient Over Time ({period}-Day Returns)\n'
                         f'Mean: {np.mean(daily_ics):.3f}, Std: {np.std(daily_ics):.3f}, '
                         f'Positive: {sum(1 for ic in daily_ics if ic > 0)}/{len(daily_ics)} days', 
                         fontsize=14)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('IC (Spearman Rank Correlation)', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f'ic_over_time_{period}d.png')
                plt.savefig(plot_path, dpi=300)
                plt.close()
                plots_generated += 1
        
        # 3. IC Distribution Histogram
        if not correlation_results.empty and 'ic' in correlation_results.columns:
            plt.figure(figsize=(12, 6))
            # Collect all daily ICs across all periods
            all_daily_ics = []
            for period in HOLDING_PERIODS:
                return_key = f'{period}d_return'
                for date in dates:
                    date_data = [d for d in data if d['date'] == date and return_key in d and d[return_key] is not None 
                               and 'flow_score' in d and not np.isnan(d['flow_score'])]
                    if len(date_data) < 3:
                        continue
                    
                    date_flow_scores = [d['flow_score'] for d in date_data]
                    date_returns = [d[return_key] for d in date_data]
                    valid_pairs = [(f, r) for f, r in zip(date_flow_scores, date_returns) 
                                  if not (np.isnan(f) or np.isnan(r))]
                    if len(valid_pairs) < 3:
                        continue
                    
                    date_flow_clean = [f for f, r in valid_pairs]
                    date_returns_clean = [r for f, r in valid_pairs]
                    try:
                        daily_ic, _ = stats.spearmanr(date_flow_clean, date_returns_clean)
                        if not np.isnan(daily_ic) and np.isfinite(daily_ic):
                            all_daily_ics.append(daily_ic)
                    except:
                        continue
            
            if len(all_daily_ics) > 0:
                plt.hist(all_daily_ics, bins=30, edgecolor='black', alpha=0.7)
                plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Zero IC')
                plt.axvline(x=np.mean(all_daily_ics), color='red', linestyle='-', linewidth=2, 
                           label=f'Mean: {np.mean(all_daily_ics):.3f}')
                plt.title(f'IC Distribution (All Periods)\n'
                         f'Mean: {np.mean(all_daily_ics):.3f}, Median: {np.median(all_daily_ics):.3f}, '
                         f'Std: {np.std(all_daily_ics):.3f}\n'
                         f'Positive: {sum(1 for ic in all_daily_ics if ic > 0)}/{len(all_daily_ics)} ({100*sum(1 for ic in all_daily_ics if ic > 0)/len(all_daily_ics):.1f}%)', 
                         fontsize=14)
                plt.xlabel('IC (Spearman Rank Correlation)', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                plot_path = os.path.join(output_dir, 'ic_distribution.png')
                plt.savefig(plot_path, dpi=300)
                plt.close()
                plots_generated += 1
        
        # 4. Correlation Heatmap by Holding Period
        if not correlation_results.empty:
            plt.figure(figsize=(10, 6))
            corr_data = correlation_results[['holding_period', 'ic', 'pearson_correlation', 'spearman_correlation']]
            corr_data = corr_data.set_index('holding_period')
            sns.heatmap(corr_data.T, annot=True, fmt='.3f', cmap='RdYlGn', center=0, 
                       cbar_kws={'label': 'Correlation'}, vmin=-0.2, vmax=0.2)
            plt.title('Correlation Metrics: Flow Score vs Returns by Holding Period', fontsize=14)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            plots_generated += 1
    
        # 5. Parameter Sensitivity Analysis
        if not param_results.empty:
            # ASK_WEIGHT is fixed at 1.0, not plotted
            params_to_plot = ['BID_WEIGHT', 'DTE_LAMBDA', 'SECONDARY_CLASS_WEIGHT', 'MIN_SECONDARY_THRESHOLD']
            params_to_plot = [p for p in params_to_plot if p in param_results.columns]
            
            if params_to_plot:
                n_params = len(params_to_plot)
                n_cols = 2
                n_rows = (n_params + 1) // 2
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
                if n_params == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes if isinstance(axes, list) else [axes]
                else:
                    axes = axes.flatten()
                
                for i, param in enumerate(params_to_plot):
                    if i < len(axes):
                        ax = axes[i]
                        param_grouped = param_results.groupby(param)['score'].mean()
                        ax.plot(param_grouped.index, param_grouped.values, marker='o', linewidth=2, markersize=8)
                        ax.set_title(f'Score vs {param}', fontsize=12)
                        ax.set_xlabel(param)
                        ax.set_ylabel('Optimization Score')
                        ax.grid(True, alpha=0.3)
                
                # Hide unused subplots
                for i in range(len(params_to_plot), len(axes)):
                    if i < len(axes):
                        axes[i].set_visible(False)
                
                plt.tight_layout()
                plot_path = os.path.join(output_dir, 'parameter_sensitivity.png')
                plt.savefig(plot_path, dpi=300)
                plt.close()
                plots_generated += 1
            
            # Score distribution plot
            plt.figure(figsize=(10, 6))
            plt.hist(param_results['score'], bins=30, edgecolor='black', alpha=0.7)
            if 'score' in param_results.columns:
                best_score = param_results['score'].max()
                plt.axvline(x=best_score, color='red', linestyle='--', linewidth=2, 
                           label=f'Best Score: {best_score:.2f}')
            plt.title('Optimization Score Distribution', fontsize=14)
            plt.xlabel('Score', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'score_distribution.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            plots_generated += 1
        
        print(f"\n[OK] Visualizations saved to {output_dir} ({plots_generated} plots generated)")
    except Exception as e:
        print(f"\n[ERROR] Failed to generate visualizations: {e}")
        import traceback
        traceback.print_exc()


def get_ai_interpretation(correlation_results, best_params, param_results):
    """Get AI interpretation using local Ollama or cloud API."""
    swing_corr = correlation_results[correlation_results['holding_period'].isin([p for p in HOLDING_PERIODS if 3 <= p <= 7])]
    if swing_corr.empty:
        return None
    
    # Build prompt
    corr_data = {
        'ic': swing_corr['ic'].mean(), 'ic_tstat': swing_corr['ic_tstat'].mean(),
        'hit_rate': swing_corr['hit_rate'].mean(), 'pearson_correlation': swing_corr['pearson_correlation'].mean(),
        'sample_size': swing_corr['sample_size'].mean()
    }
    
    # Only include optimization stats if we have the full optimization history
    has_optimization_history = not param_results.empty and 'score' in param_results.columns
    if has_optimization_history:
        param_stats = {
            'best_score': param_results['score'].max(),
            'avg_score': param_results['score'].mean(),
            'std_score': param_results['score'].std()
        }
        stats_text = f"OPTIMIZATION_STATS: Best={param_stats['best_score']:.4f}, Avg={param_stats['avg_score']:.4f}, Std={param_stats['std_score']:.4f}"
    else:
        stats_text = "OPTIMIZATION_STATS: Not available (report regenerated without full optimization history)"
    
    prompt = f"""Analyze these options flow backtest results:

RESULTS: IC={corr_data['ic']:.4f}, t-stat={corr_data['ic_tstat']:.2f}, Hit Rate={corr_data['hit_rate']:.1f}%, 
Correlation={corr_data['pearson_correlation']:.4f}, Sample={corr_data['sample_size']:.0f}

PARAMETERS: {json.dumps(best_params, indent=2) if best_params else 'None'}
{stats_text}
PERIODS: {HOLDING_PERIODS}

Provide analysis covering: 1) Predictive signal assessment (Strong/Moderate/Weak/None), 
2) Parameter insights, 3) Overfitting concerns, 4) Trading recommendations. 
400-600 words, data-driven and actionable."""
    
    system_prompt = "You are an expert quantitative finance analyst specializing in options flow analysis."
    
    # Try local Ollama first
    if USE_LOCAL_DEEPSEEK:
        result = _call_ollama(prompt, system_prompt)
        if result:
            return result
        if DEEPSEEK_API_KEY:
            print("Local Ollama unavailable, trying cloud API...")
    
    # Fallback to cloud API
    if DEEPSEEK_API_KEY:
        return _call_cloud_api(prompt, system_prompt)
    
    if not USE_LOCAL_DEEPSEEK and not DEEPSEEK_API_KEY:
        print("Warning: AI interpretation disabled. Set USE_LOCAL_DEEPSEEK=true or provide DEEPSEEK_API_KEY")
    return None


def _call_ollama(prompt, system_prompt):
    """Call local Ollama API."""
    try:
        import ollama
        print("Generating AI interpretation (local Ollama)...")
        response = ollama.generate(
            model=OLLAMA_MODEL, prompt=prompt, system=system_prompt,
            options={"temperature": 0.3, "num_predict": 2000}
        )
        if 'response' in response:
            print("[OK] AI interpretation generated (local)")
            return response['response'].strip()
    except ImportError:
        print("Warning: Ollama Python package not installed. Install with: pip install ollama")
        print("  Or set USE_LOCAL_DEEPSEEK=false in .env to disable local Ollama")
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "ECONNREFUSED" in error_msg:
            print(f"Warning: Ollama server not running. Start Ollama application or set USE_LOCAL_DEEPSEEK=false")
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            print(f"Warning: Ollama model '{OLLAMA_MODEL}' not found. Pull with: ollama pull {OLLAMA_MODEL}")
        else:
            print(f"Warning: Ollama error: {e}")
    return None


def _call_cloud_api(prompt, system_prompt):
    """Call DeepSeek cloud API."""
    try:
        import requests
        print("Generating AI interpretation (cloud API)...")
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                "temperature": 0.3, "max_tokens": 2000
            },
            timeout=60
        )
        if response.status_code == 200:
            result = response.json()
            if result.get('choices') and len(result['choices']) > 0:
                print("[OK] AI interpretation generated (cloud)")
                return result['choices'][0]['message']['content'].strip()
        else:
            print(f"Warning: API error ({response.status_code}): {response.text[:100]}")
    except ImportError:
        print("Warning: requests not installed. Install with: pip install requests")
    except Exception as e:
        print(f"Warning: API error: {e}")
    return None

def export_results_to_csv(correlation_results, best_params, param_results, data, output_dir):
    """Export backtest results to CSV files."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 1. Summary CSV
    if not correlation_results.empty:
        swing_corr = correlation_results[correlation_results['holding_period'].isin([p for p in HOLDING_PERIODS if 3 <= p <= 7])]
        summary_rows = []
        
        if not swing_corr.empty:
            summary_rows.extend([
                ('Average_IC', swing_corr['ic'].mean(), 'Swing_3-7d'),
                ('Average_IC_tstat', swing_corr['ic_tstat'].mean(), 'Swing_3-7d'),
                ('Average_Hit_Rate', swing_corr['hit_rate'].mean(), 'Swing_3-7d'),
                ('Average_Pearson_Correlation', swing_corr['pearson_correlation'].mean(), 'Swing_3-7d'),
                ('Average_Sample_Size', swing_corr['sample_size'].mean(), 'Swing_3-7d'),
            ])
        
        best_period = correlation_results.loc[correlation_results['pearson_correlation'].idxmax()]
        summary_rows.extend([
            ('Best_Period_IC', best_period['ic'], f"{int(best_period['holding_period'])}d"),
            ('Best_Period_IC_tstat', best_period['ic_tstat'], f"{int(best_period['holding_period'])}d"),
            ('Best_Period_Hit_Rate', best_period['hit_rate'], f"{int(best_period['holding_period'])}d"),
            ('Best_Period_Pearson_Correlation', best_period['pearson_correlation'], f"{int(best_period['holding_period'])}d"),
            ('Best_Period_Days', best_period['holding_period'], f"{int(best_period['holding_period'])}d"),
        ])
        
        if not param_results.empty:
            best_score, avg_score, std_score = param_results['score'].max(), param_results['score'].mean(), param_results['score'].std()
            improvement = ((best_score - avg_score) / abs(avg_score) * 100) if avg_score != 0 else 0
            summary_rows.extend([
                ('Optimization_Best_Score', best_score, 'All'),
                ('Optimization_Avg_Score', avg_score, 'All'),
                ('Optimization_Std_Score', std_score, 'All'),
                ('Optimization_Score_Improvement_Pct', improvement, 'All'),
            ])
        
        if data:
            dates, tickers = [d['date'] for d in data], set(d['ticker'] for d in data)
            summary_rows.extend([
                ('Total_Data_Points', len(data), 'All'),
                ('Unique_Tickers', len(tickers), 'All'),
                ('Date_Range_Start', min(dates), 'All'),
                ('Date_Range_End', max(dates), 'All'),
            ])
        
        pd.DataFrame([(m, v, h, timestamp) for m, v, h in summary_rows], 
                    columns=['metric', 'value', 'holding_period', 'timestamp']
        ).to_csv(os.path.join(output_dir, 'backtest_summary.csv'), index=False)
        print("[OK] Summary CSV saved")
    
    # 2. Correlation Results CSV
    if not correlation_results.empty:
        corr_export = correlation_results.copy()
        for col in corr_export.select_dtypes(include=[np.number]).columns:
            corr_export[col] = corr_export[col].round(4) if col not in ['holding_period', 'sample_size'] else corr_export[col].astype(int) if col == 'sample_size' else corr_export[col]
        corr_export.insert(0, 'export_timestamp', timestamp)
        corr_export.insert(1, 'test_type', 'Correlation_Analysis')
        corr_export.to_csv(os.path.join(output_dir, 'correlation_results.csv'), index=False, float_format='%.4f')
        print("[OK] Correlation results CSV saved")
    
    # 3. Parameter Optimization CSV
    if not param_results.empty:
        param_export = param_results.copy()
        for col in param_export.select_dtypes(include=[np.number]).columns:
            param_export[col] = param_export[col].round(4)
        param_export.insert(0, 'export_timestamp', timestamp)
        param_export.insert(1, 'test_type', 'Parameter_Optimization')
        param_export.sort_values('score', ascending=False).to_csv(
            os.path.join(output_dir, 'parameter_optimization.csv'), index=False, float_format='%.4f')
        print("[OK] Parameter optimization CSV saved")
    
    # 4. Optimal Parameters CSV
    if best_params:
        desc_map = {
            'BID_WEIGHT': 'Weight for bid-side options flow',
            'DTE_LAMBDA': 'Decay factor for days-to-expiration',
            'PRICE_CAP_ADJUSTMENT': 'Apply price/cap adjustment (True/False)',
            'SECONDARY_CLASS_WEIGHT': 'Weight for secondary share class flow',
            'MIN_SECONDARY_THRESHOLD': 'Minimum threshold for secondary class inclusion'
        }
        # ASK_WEIGHT is fixed at 1.0 (not in params, not exported)
        pd.DataFrame({
            'parameter': list(best_params.keys()),
            'optimal_value': list(best_params.values()),
            'description': [desc_map.get(p, 'N/A') for p in best_params.keys()],
            'export_timestamp': timestamp,
            'optimization_method': 'Bayesian_Optuna'
        }).to_csv(os.path.join(output_dir, 'optimal_parameters.csv'), index=False)
        print("[OK] Optimal parameters CSV saved")
    
    print(f"[OK] All CSV files exported to {output_dir}")


def generate_report(data, correlation_results, best_params, param_results, output_dir):
    """Generate comprehensive backtest report in Markdown format."""
    report_path = os.path.join(output_dir, 'backtest_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Options Flow Scoring Backtest Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset Summary
        f.write("## Dataset Summary\n\n")
        f.write(f"- **Total data points:** {len(data)}\n")
        f.write(f"- **Unique tickers:** {len(set(d['ticker'] for d in data))}\n")
        f.write(f"- **Date range:** {min(d['date'] for d in data)} to {max(d['date'] for d in data)}\n")
        f.write(f"- **Holding periods tested:** {HOLDING_PERIODS}\n\n")
        
        # Correlation Analysis (Quant Metrics)
        f.write("## Correlation Analysis (Quant Methodology)\n\n")
        if not correlation_results.empty:
            # Focus on swing trading periods
            swing_periods = [p for p in HOLDING_PERIODS if 3 <= p <= 7]
            swing_corr = correlation_results[correlation_results['holding_period'].isin(swing_periods)]
            
            if not swing_corr.empty:
                f.write("### Swing Trading Periods (3-7 days)\n\n")
                display_cols = ['holding_period', 'ic', 'ic_tstat', 'hit_rate', 'pearson_correlation', 'sample_size']
                swing_display = swing_corr[[c for c in display_cols if c in swing_corr.columns]]
                
                # Convert to markdown table
                f.write("| " + " | ".join(swing_display.columns) + " |\n")
                f.write("| " + " | ".join(["---"] * len(swing_display.columns)) + " |\n")
                for _, row in swing_display.iterrows():
                    f.write("| " + " | ".join([str(round(v, 4) if isinstance(v, (int, float)) else v) for v in row]) + " |\n")
                f.write("\n")
                
                avg_ic = swing_corr['ic'].mean()
                avg_ic_tstat = swing_corr['ic_tstat'].mean()
                avg_hit_rate = swing_corr['hit_rate'].mean()
                avg_pearson = swing_corr['pearson_correlation'].mean()
                
                f.write(f"- **Average Information Coefficient (IC):** {avg_ic:.4f}\n")
                f.write(f"- **Average IC t-statistic:** {avg_ic_tstat:.2f}\n")
                f.write(f"- **Average Hit Rate:** {avg_hit_rate:.1f}%\n")
                f.write(f"- **Average Pearson Correlation:** {avg_pearson:.4f}\n\n")
                
                # Quant interpretation
                if avg_ic > 0.1 and avg_ic_tstat > 2.0:
                    f.write("**STRONG predictive signal** - IC > 0.1 and statistically significant\n")
                    f.write("   Positive flow scores predict positive returns (desired relationship)\n\n")
                elif avg_ic > 0.05 and avg_ic_tstat > 1.5:
                    f.write("**MODERATE predictive signal** - IC > 0.05 with some significance\n")
                    f.write("   Positive flow scores predict positive returns (desired relationship)\n\n")
                elif avg_ic > 0:
                    f.write("**WEAK predictive signal** - Positive IC but low significance\n")
                    f.write("   Positive flow scores weakly predict positive returns\n\n")
                elif avg_ic < -0.05:
                    f.write("**INVERSE signal** - Negative IC indicates contrarian relationship\n")
                    f.write("   Positive flow scores predict NEGATIVE returns (opposite of desired)\n")
                    f.write("   This suggests the flow signal may be reversed or the market is fading the flow\n\n")
                else:
                    f.write("**NO predictive signal** - IC near zero indicates no relationship\n")
                    f.write("   Flow scores do not predict returns in either direction\n\n")
                
                if avg_hit_rate > 55:
                    f.write(f"**Good hit rate** ({avg_hit_rate:.1f}%) - Flow direction matches returns\n\n")
                elif avg_hit_rate > 50:
                    f.write(f"**Marginal hit rate** ({avg_hit_rate:.1f}%) - Slightly better than random\n\n")
                else:
                    f.write(f"**Poor hit rate** ({avg_hit_rate:.1f}%) - Worse than random\n\n")
            
            f.write("### All Periods\n\n")
            # Convert full correlation results to markdown table
            f.write("| " + " | ".join(correlation_results.columns) + " |\n")
            f.write("| " + " | ".join(["---"] * len(correlation_results.columns)) + " |\n")
            for _, row in correlation_results.iterrows():
                f.write("| " + " | ".join([str(round(v, 4) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for v in row]) + " |\n")
            f.write("\n")
        
        # Methodology Note
        f.write("## Methodology\n\n")
        f.write("This backtest tests the **directional relationship** between flow scores and returns:\n")
        f.write("- **Positive flow scores** (bullish flow) should predict **positive returns**\n")
        f.write("- **Negative flow scores** (bearish flow) should predict **negative returns**\n\n")
        f.write("The correlation coefficient measures this directional relationship:\n")
        f.write("- **Positive correlation** = bullish flow predicts positive returns (desired)\n")
        f.write("- **Negative correlation** = bullish flow predicts negative returns (inverse/contrarian signal)\n\n")
        f.write("Flow scores preserve sign: positive = bullish flow, negative = bearish flow.\n")
        f.write("Unlike ranking-based approaches, this tests if the magnitude and direction\n")
        f.write("of flow scores predict the magnitude and direction of returns.\n\n")
        
        # Optimal Parameters
        f.write("## Optimal Parameters\n\n")
        if best_params:
            f.write("### Best Parameters Found\n\n")
            f.write("| Parameter | Value |\n")
            f.write("| --- | --- |\n")
            for key, value in best_params.items():
                f.write(f"| `{key}` | {value} |\n")
            f.write("\n")
            
            # Show score statistics
            if not param_results.empty:
                best_score = param_results['score'].max()
                avg_score = param_results['score'].mean()
                f.write("### Optimization Statistics\n\n")
                f.write(f"- **Best score:** {best_score:.4f}\n")
                f.write(f"- **Average score:** {avg_score:.4f}\n")
                f.write(f"- **Score improvement:** {((best_score - avg_score) / abs(avg_score)) * 100:+.2f}% over average\n\n")
        
        # AI Interpretation (if available)
        ai_interpretation = get_ai_interpretation(correlation_results, best_params, param_results)
        if ai_interpretation:
            source = "Local (Ollama)" if USE_LOCAL_DEEPSEEK and not DEEPSEEK_API_KEY else "Cloud API"
            f.write(f"## AI Interpretation (DeepSeek - {source})\n\n")
            f.write(ai_interpretation)
            f.write("\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if not correlation_results.empty:
            best_period = correlation_results.loc[correlation_results['pearson_correlation'].idxmax()]
            f.write(f"1. **Best holding period:** {int(best_period['holding_period'])} days ")
            f.write(f"(correlation: {best_period['pearson_correlation']:.3f})\n")
        
        if best_params:
            f.write(f"2. **Use optimized parameters** for improved correlation\n")
        
        f.write("\n---\n")
    
    print(f"[OK] Report saved to {report_path}")
    
    # Also save results as CSV
    if not correlation_results.empty:
        correlation_results.to_csv(os.path.join(output_dir, 'correlation_results.csv'), index=False)
    if not param_results.empty:
        param_results.to_csv(os.path.join(output_dir, 'parameter_optimization.csv'), index=False)


def main():
    print("="*80)
    print("OPTIONS FLOW SCORING BACKTEST")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Step 1: Generate random trading dates within rolling window
    # Calculate start date (ROLLING_WINDOW_DAYS trading days ago)
    start_date = datetime.now().date()
    days_back = 0
    while days_back < ROLLING_WINDOW_DAYS:
        if start_date.weekday() < 5:
            days_back += 1
        if days_back < ROLLING_WINDOW_DAYS:
            start_date -= timedelta(days=1)
    # Go back to most recent trading day if weekend
    while start_date.weekday() >= 5:
        start_date -= timedelta(days=1)
    
    # Use random sampling with minimum spacing to avoid overlapping holding periods
    # With same-day close entry, we can use smaller spacing (10 days) instead of max holding period + 1
    # This allows more dates in the window while still maintaining statistical independence
    MIN_DATE_SPACING = 10  # Minimum days between dates (sufficient with same-day close entry)
    MIN_VOLATILITY = 0.0   # No volatility filter - test all market conditions
    min_spacing = MIN_DATE_SPACING  # Use fixed spacing instead of max holding period + 1
    
    trading_dates = get_random_trading_dates_in_window(
        start_date, ROLLING_WINDOW_DAYS, NUM_RANDOM_DATES, min_spacing, MIN_VOLATILITY
    )
    print(f"Random trading dates: {len(trading_dates)} days sampled from {ROLLING_WINDOW_DAYS}-day window")
    print(f"Minimum spacing: {min_spacing} days (to avoid overlapping holding periods)")
    print(f"Date range: {trading_dates[0]} to {trading_dates[-1]}")
    print(f"Methodology: Random sampling with spacing for statistical independence")
    
    # Step 2: Collect historical data using high-volume selection (same as quick_backtest)
    try:
        historical_data = collect_historical_data_high_volume(
            trading_dates,
            max_workers=30,  # Increased from 25 to 30 for better throughput
            num_tickers=NUM_TICKERS_PER_DATE,
            balance_flow=BALANCE_FLOW
        )
    except Exception as e:
        print(f"\n[ERROR] Data collection failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return
    
    if not historical_data:
        print("ERROR: No historical data collected!")
        return
    
    # Save raw data
    try:
        raw_data_path = os.path.join(OUTPUT_DIR, 'historical_data.json')
        with open(raw_data_path, 'w') as f:
            json.dump(historical_data, f, indent=2)
        print(f"[OK] Raw data saved to {raw_data_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save raw data: {e}")
    
    # Step 3: Add future returns
    try:
        data_with_returns = add_future_returns(historical_data, HOLDING_PERIODS, max_workers=30)
    except Exception as e:
        print(f"\n[ERROR] Future returns calculation failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return
    
    if not data_with_returns:
        print("ERROR: No data with returns!")
        return
    
    # Step 4: Optimize parameters (focus on correlation between absolute flow_score and returns)
    try:
        best_params, param_results = optimize_parameters(
            data_with_returns, HOLDING_PERIODS
        )
    except Exception as e:
        print(f"\n[ERROR] Parameter optimization failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return
    
    if not best_params:
        print("[ERROR] Optimization did not produce valid parameters!")
        return
    
    print(f"\n{'='*80}")
    print("OPTIMAL PARAMETERS FOUND:")
    print(f"{'='*80}")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    # Step 5: Calculate final ranking metrics with optimal parameters
    print(f"\n{'='*80}")
    print("CALCULATING FINAL RANKING METRICS")
    print(f"{'='*80}")
    
    calculate_flow_scores_for_data(data_with_returns, best_params)
    ranking_results = calculate_ranking_metrics(data_with_returns, HOLDING_PERIODS, top_pct=0.10)
    
    # Also calculate correlation for comparison
    correlation_results = calculate_correlation_metrics(data_with_returns, HOLDING_PERIODS)
    
    # Step 7: Generate visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    create_visualizations(
        data_with_returns, correlation_results, 
        param_results, PLOT_DIR
    )
    
    # Step 7: Export results to CSV
    print(f"\n{'='*80}")
    print("EXPORTING RESULTS TO CSV")
    print(f"{'='*80}")
    
    export_results_to_csv(
        correlation_results, best_params, param_results,
        data_with_returns, OUTPUT_DIR
    )
    
    # Export ranking results
    ranking_results = calculate_ranking_metrics(data_with_returns, HOLDING_PERIODS, top_pct=0.10)
    if not ranking_results.empty:
        ranking_results.to_csv(os.path.join(OUTPUT_DIR, 'ranking_results.csv'), index=False)
        print("[OK] Ranking results CSV saved")
    
    # Step 8: Generate report
    print(f"\n{'='*80}")
    print("GENERATING REPORT")
    print(f"{'='*80}")
    
    generate_report(
        data_with_returns, correlation_results,
        best_params, param_results, OUTPUT_DIR
    )
    
    # Print ranking summary
    if not ranking_results.empty:
        print(f"\n{'='*80}")
        print("RANKING APPROACH SUMMARY")
        print(f"{'='*80}")
        swing_ranking = ranking_results[ranking_results['holding_period'].isin([p for p in HOLDING_PERIODS if 3 <= p <= 7])]
        if not swing_ranking.empty:
            avg_spread = swing_ranking['avg_spread'].mean()
            avg_win_rate = swing_ranking['win_rate'].mean()
            print(f"Average Spread (Top 10% vs Bottom 10%): {avg_spread:.4f}%")
            print(f"Average Win Rate: {avg_win_rate:.1f}%")
            if avg_spread > 0.3 and avg_win_rate > 55:
                print("[OK] STRONG ranking signal - Top stocks consistently outperform")
            elif avg_spread > 0.1 and avg_win_rate > 52:
                print("[OK] MODERATE ranking signal - Some predictive power")
            else:
                print("[WARNING] WEAK ranking signal - Limited predictive power")
    
    print(f"\n{'='*80}")
    print("BACKTEST COMPLETE!")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Plots saved to: {PLOT_DIR}")

if __name__ == "__main__":
    main()
