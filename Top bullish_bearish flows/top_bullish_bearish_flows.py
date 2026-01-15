import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import numpy as np
import pytz
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure stdout for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except:
        pass
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# Load environment and initialize Massive API
# Look for .env in parent Python Scripts directory (2 levels up)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)
API_KEY = os.getenv("MASSIVE_API_KEY")
if not API_KEY:
    print(f"ERROR: MASSIVE_API_KEY not found in .env file!")
    print(f"  Looked for .env at: {env_path}")
    exit(1)

from massive import RESTClient
massive_client = RESTClient(API_KEY)

# ============================================================================
# CONFIGURATION - Customize these settings
# ============================================================================

# File Paths
OUTPUT_DIR = "CSV_Output"
# Load tickers from filtered list (stocks with significant options flow)
FILTERED_TICKERS_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "Optionable Stocks", "stocks_with_significant_options_flow.csv")

# Performance Settings
MAX_CONTRACTS_PER_TICKER = 300  # Top N contracts by volume (captures 95%+ of flow)
MAX_WORKERS = 100               # Parallel processing threads (increase for production servers)
MIN_VOLUME_FILTER = 1           # Minimum volume to include a contract

# Output Settings
TOP_N_STOCKS = 20               # Number of top bullish/bearish stocks to return (overall)
TOP_N_ETFS = 10                 # Number of top bullish/bearish ETFs to return
TOP_N_PER_CATEGORY = 10         # Number per market cap category (mega/large/mid/small)

# ETF Filters
MIN_ETF_VOLUME = 100_000        # Minimum total options volume for ETFs (filters low-liquidity like VOO)

# Weighting Parameters
ASK_WEIGHT = 1.0                # Weight for aggressive trades (ask side)
BID_WEIGHT = 0.7                # Weight for passive trades (bid side)
DTE_LAMBDA = 0.10               # Exponential decay rate for DTE weighting
VOLATILITY_WEIGHT = 0.5         # Weight for volatility normalization (higher = more penalty for volatile stocks)

# Multi-Class Flow Aggregation (liquidity-weighted approach)
MULTI_CLASS_AGGREGATION = True  # Enable flow aggregation across share classes
SECONDARY_CLASS_WEIGHT = 0.30   # Weight for secondary class flow (30% of secondary flow added to primary)
MIN_SECONDARY_THRESHOLD = 0.20  # Secondary class must have ≥20% of primary's flow to count

# Share Class Groups (for duplicate removal)
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

# Timezone
et_tz = pytz.timezone('US/Eastern')

# Volatility Settings
# For swing trading (1-5 day holds): Use 10-20 days (more responsive to recent conditions)
# For longer-term positions: Use 30-60 days (academic standard)
VOLATILITY_LOOKBACK_DAYS = 10  # Recommended: 10 for very responsive, 20 for balanced, 30 for standard
VOLATILITY_CACHE_FILE = os.path.join(OUTPUT_DIR, "volatility_cache.json")
VOLATILITY_CACHE_HOURS = 6  # Refresh every 6 hours

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_analysis_date():
    """Get the appropriate date for analysis (handles weekends/market hours)."""
    now_et = datetime.now(et_tz)
    date = now_et.date()
    
    if date.weekday() >= 5:  # Weekend
        while date.weekday() >= 5:
            date -= timedelta(days=1)
    elif now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 30):  # Before market open
        date -= timedelta(days=1)
        while date.weekday() >= 5:
            date -= timedelta(days=1)
    
    return date.strftime("%Y-%m-%d")

def load_filtered_tickers():
    """Load tickers from filtered list (stocks with significant options flow)."""
    try:
        if not os.path.exists(FILTERED_TICKERS_CSV):
            print(f"ERROR: Filtered tickers CSV not found at {FILTERED_TICKERS_CSV}")
            print("Please run stocks_with_significant_options_flow.py first to generate the list.")
            return []
        
        df = pd.read_csv(FILTERED_TICKERS_CSV)
        tickers = df['Ticker'].dropna().str.strip().str.upper().tolist()
        print(f"Loaded {len(tickers)} tickers from filtered list")
        return sorted(tickers)
    except Exception as e:
        print(f"Error loading filtered tickers CSV: {e}")
        return []

def load_volatility_cache():
    """Load cached volatility data if fresh."""
    try:
        if os.path.exists(VOLATILITY_CACHE_FILE):
            cache = pd.read_json(VOLATILITY_CACHE_FILE)
            cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
            if datetime.now() - cache_time < timedelta(hours=VOLATILITY_CACHE_HOURS):
                return cache.get('data', {})
    except:
        pass
    return {}

def save_volatility_cache(volatilities):
    """Save volatility data to cache."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pd.Series({'timestamp': datetime.now().isoformat(), 'data': volatilities}).to_json(VOLATILITY_CACHE_FILE)
    except:
        pass

def calculate_volatility(ticker, cache=None):
    """Calculate historical volatility (optimized with caching)."""
    # Check cache first
    if cache and ticker in cache:
        return cache[ticker]
    
    try:
        end_date = datetime.now().date()
        # Fetch enough days to account for weekends (add 40% buffer)
        start_date = end_date - timedelta(days=int(VOLATILITY_LOOKBACK_DAYS * 1.4))
        
        # Fetch price data for volatility calculation
        aggs = list(massive_client.list_aggs(
            ticker, 1, "day", 
            start_date.strftime("%Y-%m-%d"), 
            end_date.strftime("%Y-%m-%d"), 
            limit=VOLATILITY_LOOKBACK_DAYS
        ))
        
        if len(aggs) < 5:
            return 0.30
        
        # Extract closes efficiently using numpy
        closes = np.array([float(agg.close) for agg in aggs if hasattr(agg, 'close')])
        
        if len(closes) < 5:
            return 0.30
        
        # Calculate returns using numpy (vectorized, much faster)
        returns = np.diff(closes) / closes[:-1]
        returns = returns[np.isfinite(returns)]  # Remove inf/nan
        
        if len(returns) < 3:
            return 0.30
        
        # Annualize volatility (252 trading days)
        volatility = np.std(returns) * np.sqrt(252)
        
        # Clamp to reasonable range
        return float(np.clip(volatility, 0.05, 2.0))
    except:
        return 0.30

def calculate_flow_score(net_flow, total_volume, market_cap, volatility, stock_price=None, is_etf=False, aum=None):
    """
    Calculate unified flow score with stock price bias correction and volatility normalization.
    Formula: 
      - Stocks: net_flow / (log10(market_cap) × price_cap_adjustment × volatility_adjustment)
      - ETFs: net_flow / log10(AUM) - normalized by assets under management
    """
    # For ETFs, use AUM-based scoring
    if is_etf:
        if aum and aum > 0:
            aum_log = np.log10(max(aum, 1e6))
            return net_flow / aum_log
        else:
            # Fallback to simple net_flow if AUM not available
            return net_flow
    
    # For stocks, require valid market cap
    if not market_cap or market_cap <= 0:
        return 0.0
    
    # Market cap normalization (log scaling)
    market_cap_log = np.log10(max(market_cap, 1e6))
    
    # Stock price bias correction (penalizes expensive stocks like BKNG)
    if stock_price and stock_price > 0:
        market_cap_billions = market_cap / 1e9
        price_cap_ratio = stock_price / market_cap_billions
        price_cap_adjustment = max(1.0, price_cap_ratio)  # Only penalize if ratio > 1
    else:
        price_cap_adjustment = 1.0
    
    # Volatility normalization (penalizes high volatility stocks)
    # Higher volatility stocks naturally have larger option flows, so normalize by volatility
    if volatility and volatility > 0:
        volatility_adjustment = 1.0 + (volatility * VOLATILITY_WEIGHT)
    else:
        volatility_adjustment = 1.0
    
    return net_flow / (market_cap_log * price_cap_adjustment * volatility_adjustment)

# ============================================================================
# MAIN DATA FETCHING
# ============================================================================

KNOWN_ETFS = {
    'SPY', 'QQQ', 'IWM', 'DIA', 'EEM', 'FXI', 'GLD', 'SLV', 'TLT', 'HYG', 'LQD',
    'XLF', 'XLE', 'XLK', 'XLV', 'XLP', 'XLI', 'XLU', 'XLB', 'XLY', 'XLRE', 'XLC',
    'VXX', 'UVXY', 'SVXY', 'SQQQ', 'TQQQ', 'SPXL', 'SPXS', 'UPRO', 'SPXU',
    'EWJ', 'EWZ', 'EWW', 'EWG', 'EWU', 'EWC', 'EWA', 'EWH', 'EWY', 'EWT',
    'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'BND', 'VNQ', 'VYM', 'VIG', 'VUG',
    'ARKK', 'ARKG', 'ARKW', 'ARKF', 'ARKQ',
    'SMH', 'SOXX', 'XBI', 'IBB', 'XOP', 'XME', 'ITB', 'XHB', 'KRE', 'KBE',
    'GDXJ', 'GDX', 'SLX', 'USO', 'UNG', 'DBO', 'DBB',
    'TAN', 'ICLN', 'PBW', 'QCLN', 'FAN',
    'JETS', 'AWAY', 'XRT', 'XLF', 'XHB'
}

def get_market_cap_category(market_cap):
    """Categorize stock by market cap."""
    if market_cap >= 200_000_000_000:
        return 'mega_cap'
    elif market_cap >= 10_000_000_000:
        return 'large_cap'
    elif market_cap >= 2_000_000_000:
        return 'mid_cap'
    return 'small_cap'

def calculate_contract_flow(contract):
    """Calculate bullish/bearish flow for a single contract."""
    day = getattr(contract, 'day', None)
    if not day:
        return None
    
    volume = getattr(day, 'volume', 0) or 0
    if volume < MIN_VOLUME_FILTER:
        return None
    
    details = getattr(contract, 'details', None)
    if not details:
        return None
    
    option_type = getattr(details, 'contract_type', '').upper()
    expiry_str = getattr(details, 'expiration_date', None)
    close_price = getattr(day, 'close', 0) or 0
    last_trade = getattr(contract, 'last_trade', None)
    last_price = getattr(last_trade, 'price', close_price) if last_trade else close_price
    
    if not last_price:
        return None
    
    premium = close_price * volume * 100
    bid = getattr(day, 'open', close_price * 0.98) or close_price * 0.98
    ask = getattr(day, 'close', close_price * 1.02) or close_price * 1.02
    mid = (bid + ask) / 2
    ask_side_pct, bid_side_pct = (0.7, 0.3) if last_price >= mid else (0.3, 0.7)
    
    # DTE weight
    if expiry_str:
        try:
            expiry = datetime.strptime(expiry_str, '%Y-%m-%d').date()
            dte = (expiry - datetime.now().date()).days
            import math
            dte_weight = max(0.1, math.exp(-DTE_LAMBDA * dte))
        except:
            dte_weight = 1.0
    else:
        dte_weight = 1.0
    
    # Calculate flows
    if option_type == 'CALL':
        bullish = premium * ask_side_pct * ASK_WEIGHT * dte_weight
        bearish = premium * bid_side_pct * BID_WEIGHT * dte_weight
    else:  # PUT
        bearish = premium * ask_side_pct * ASK_WEIGHT * dte_weight
        bullish = premium * bid_side_pct * BID_WEIGHT * dte_weight
    
    underlying_price = getattr(details, 'underlying_price', None) or getattr(contract, 'underlying_price', None)
    
    return {
        'bullish': bullish,
        'bearish': bearish,
        'volume': volume,
        'underlying_price': underlying_price
    }

def fetch_ticker_data(ticker):
    """Fetch market cap and options data for a ticker."""
    try:
        is_etf = ticker in KNOWN_ETFS
        details = massive_client.get_ticker_details(ticker)
        market_cap = getattr(details, 'market_cap', 0)
        
        if not is_etf and (not market_cap or market_cap < 300_000_000):
            return None
        
        shares_outstanding = getattr(details, 'weighted_shares_outstanding', 0) or \
                           getattr(details, 'share_class_shares_outstanding', 0)
        stock_price = (market_cap / shares_outstanding) if shares_outstanding and market_cap else None
        category = 'etf' if is_etf else get_market_cap_category(market_cap)
        
        # Get and process options contracts
        all_contracts = list(massive_client.list_snapshot_options_chain(ticker))
        if not all_contracts:
            return None
        
        sorted_contracts = sorted(
            all_contracts,
            key=lambda c: getattr(getattr(c, 'day', None), 'volume', 0) or 0,
            reverse=True
        )[:MAX_CONTRACTS_PER_TICKER]
        
        flows = [f for f in (calculate_contract_flow(c) for c in sorted_contracts) if f]
        if not flows:
            return None
        
        total_bullish = sum(f['bullish'] for f in flows)
        total_bearish = sum(f['bearish'] for f in flows)
        total_volume = sum(f['volume'] for f in flows)
        
        if is_etf and total_volume < MIN_ETF_VOLUME:
            return None
        
        # Calculate AUM for ETFs
        aum = None
        if is_etf and shares_outstanding:
            avg_price = sum(f.get('underlying_price', 0) for f in flows if f.get('underlying_price')) / len(flows)
            if avg_price > 0:
                aum = avg_price * shares_outstanding
        
        return {
            'ticker': ticker,
            'market_cap': market_cap,
            'stock_price': stock_price,
            'category': category,
            'is_etf': is_etf,
            'aum': aum,
            'bullish_flow': total_bullish,
            'bearish_flow': total_bearish,
            'total_volume': total_volume
        }
    except:
        return None

# ============================================================================
# DATA PROCESSING
# ============================================================================

def aggregate_multi_class_flow_and_deduplicate(df):
    """Aggregate flow across share classes using liquidity-weighted approach."""
    df = df.copy()
    df['multi_class_aggregated'] = False
    
    for variants in SHARE_CLASS_GROUPS.values():
        present = [t for t in variants if t in df['ticker'].values]
        if len(present) <= 1:
            continue
            
        subset = df[df['ticker'].isin(present)].copy()
        
        for is_bullish in [True, False]:
            classes = subset[subset['net_flow'] > 0] if is_bullish else subset[subset['net_flow'] < 0]
            if len(classes) < 2:
                continue
            
            sorted_classes = classes.sort_values('flow_score', ascending=not is_bullish)
            primary_ticker = sorted_classes.iloc[0]['ticker']
            primary_flow = abs(sorted_classes.iloc[0]['flow_score'])
            
            for i in range(1, len(sorted_classes)):
                secondary_flow = abs(sorted_classes.iloc[i]['flow_score'])
                if secondary_flow >= MIN_SECONDARY_THRESHOLD * primary_flow:
                    primary_idx = df[df['ticker'] == primary_ticker].index[0]
                    added_flow = secondary_flow * SECONDARY_CLASS_WEIGHT
                    df.loc[primary_idx, 'flow_score'] += added_flow if is_bullish else -added_flow
                    df.loc[primary_idx, 'multi_class_aggregated'] = True
        
        best_ticker = subset.loc[subset['flow_score'].abs().idxmax(), 'ticker']
        df = df[~df['ticker'].isin([t for t in present if t != best_ticker])]
    
    return df

def run_parallel(func, items, desc=""):
    """Run function in parallel and return results."""
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(func, item): item for item in items}
        for i, future in enumerate(as_completed(futures), 1):
            if i % 50 == 0:
                print(f"Processed {i}/{len(items)} {desc}...")
            try:
                result = future.result()
                if result:
                    results.append(result)
            except:
                pass
    return results

def save_rankings(bullish_df, bearish_df, filename):
    """Combine bullish/bearish rankings and save to CSV."""
    data = []
    for df, flow_type in [(bullish_df, 'Bullish'), (bearish_df, 'Bearish')]:
        if not df.empty:
            df_copy = df.copy()
            df_copy['flow_type'] = flow_type
            df_copy['rank'] = range(1, len(df_copy) + 1)
            
            if 'multi_class_aggregated' in df_copy.columns:
                df_copy['ticker'] = df_copy.apply(
                    lambda row: row['ticker'] + '*' if row['multi_class_aggregated'] else row['ticker'],
                    axis=1
                )
            
            data.append(df_copy)
    if data:
        filepath = os.path.join(OUTPUT_DIR, filename)
        # Delete existing file to ensure fresh write with updated timestamp
        if os.path.exists(filepath):
            os.remove(filepath)
        
        combined = pd.concat(data, ignore_index=True)
        cols = ['rank', 'flow_type', 'ticker'] + [c for c in combined.columns if c not in ['rank', 'flow_type', 'ticker', 'multi_class_aggregated']]
        combined[cols].to_csv(filepath, index=False)
        print(f"[OK] {filepath}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("Options Flow Analysis - Top 20 Bullish/Bearish Stocks")
print(f"Analysis Date: {get_analysis_date()}")
print(f"Performance: {MAX_WORKERS} threads")

# Start performance timer
_start_time = time.time()

# Step 1: Get tickers from filtered list
print("\n[1/4] Loading filtered tickers...")
tickers = load_filtered_tickers()
print(f"Found {len(tickers)} tickers")

if not tickers:
    print("ERROR: No tickers found in filtered list. Exiting.")
    print(f"Please run stocks_with_significant_options_flow.py to generate the list.")
    exit(1)

# Step 2: Fetch market cap and options data in parallel
print(f"\n[2/4] Fetching market cap and options data (parallel: {MAX_WORKERS} workers)...")
results = run_parallel(fetch_ticker_data, tickers, "tickers")
print(f"Retrieved data for {len(results)} tickers")

if not results:
    print("ERROR: No data retrieved from any tickers.")
    print("Possible causes:")
    print("  1. API rate limiting")
    print("  2. Network issues")
    print("  3. All tickers below $300M market cap")
    exit(1)

# Step 3: Calculate volatility in parallel (with caching)
print(f"\n[3/4] Calculating volatility...")
vol_cache = load_volatility_cache()
cached_count = sum(1 for r in results if r['ticker'] in vol_cache)
if cached_count > 0:
    print(f"Using cached volatility for {cached_count}/{len(results)} tickers")

volatilities = {}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(calculate_volatility, r['ticker'], vol_cache): r['ticker'] for r in results}
    for i, future in enumerate(as_completed(futures), 1):
        ticker = futures[future]
        if i % 50 == 0:
            print(f"Calculated volatility for {i}/{len(results)} tickers...")
        volatilities[ticker] = future.result() or 0.30

save_volatility_cache(volatilities)

# Step 4: Calculate flow scores
print(f"\n[4/4] Calculating flow scores...")
for result in results:
    ticker = result['ticker']
    net_flow = result['bullish_flow'] - result['bearish_flow']
    market_cap = result['market_cap']
    volatility = volatilities.get(ticker, 0.30)
    
    result['net_flow'] = net_flow
    result['volatility'] = volatility
    result['relative_flow'] = net_flow / np.log10(market_cap) if market_cap and market_cap > 0 else 0
    result['flow_score'] = calculate_flow_score(
        net_flow, 
        result['total_volume'], 
        market_cap or 1,  # Use 1 for ETFs without market cap
        volatility, 
        stock_price=result.get('stock_price'),
        is_etf=result.get('is_etf', False),
        aum=result.get('aum')
    )

# Create DataFrame and separate stocks/ETFs
df = pd.DataFrame(results)
if 'is_etf' not in df.columns:
    df['is_etf'] = False

stocks_df = aggregate_multi_class_flow_and_deduplicate(df[df['is_etf'] == False])
etfs_df = df[df['is_etf'] == True]

# Rank by flow_score - Overall
stocks_bullish = stocks_df[stocks_df['net_flow'] > 0].nlargest(TOP_N_STOCKS, 'flow_score')
stocks_bearish = stocks_df[stocks_df['net_flow'] < 0].nsmallest(TOP_N_STOCKS, 'flow_score')
etfs_bullish = etfs_df[etfs_df['net_flow'] > 0].nlargest(TOP_N_ETFS, 'flow_score')
etfs_bearish = etfs_df[etfs_df['net_flow'] < 0].nsmallest(TOP_N_ETFS, 'flow_score')

# Rank by market cap category
category_rankings = {}
for category in ['mega_cap', 'large_cap', 'mid_cap', 'small_cap']:
    cat_df = stocks_df[stocks_df['category'] == category]
    if not cat_df.empty:
        category_rankings[category] = {
            'bullish': cat_df[cat_df['net_flow'] > 0].nlargest(TOP_N_PER_CATEGORY, 'flow_score'),
            'bearish': cat_df[cat_df['net_flow'] < 0].nsmallest(TOP_N_PER_CATEGORY, 'flow_score')
        }

# Save results to CSV
analysis_date = get_analysis_date()
print("\n" + "="*80)
print("Saving results...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save all rankings
save_rankings(stocks_bullish, stocks_bearish, f'stock_flows_{analysis_date}.csv')
save_rankings(etfs_bullish, etfs_bearish, f'etf_flows_{analysis_date}.csv')
for cat_name, rankings in category_rankings.items():
    save_rankings(rankings['bullish'], rankings['bearish'], f'{cat_name}_flows_{analysis_date}.csv')

_elapsed = time.time() - _start_time
print(f"\nAnalysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Performance: {_elapsed:.1f}s total | {_elapsed/len(results):.2f}s per ticker | {len(results)/_elapsed:.1f} tickers/sec")
