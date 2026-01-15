"""
Flat Files Loader for Massive API Historical Options Data
Downloads and parses CSV files instead of making individual API calls.
"""

import os
import boto3
import pandas as pd
import gzip
from datetime import datetime, timedelta
from dotenv import load_dotenv
from io import BytesIO
import json

# Load environment
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(env_path)

# S3 Configuration for Massive API Flat Files
POLYGON_S3_ENDPOINT = "https://files.massive.com"
POLYGON_S3_BUCKET = "flatfiles"
POLYGON_S3_ACCESS_KEY = os.getenv("POLYGON_S3_ACCESS_KEY")
POLYGON_S3_SECRET_KEY = os.getenv("POLYGON_S3_SECRET_KEY")

# Centralized cache directories (shared across all scripts)
HISTORICAL_DATA_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "historical_data")
FLAT_FILES_CACHE_DIR = os.path.join(HISTORICAL_DATA_BASE, "flat_files_cache")
AGGREGATED_CACHE_DIR = os.path.join(HISTORICAL_DATA_BASE, "aggregated_flat_files_cache")

def get_s3_client():
    """Create S3 client for Massive API Flat Files."""
    if not POLYGON_S3_ACCESS_KEY or not POLYGON_S3_SECRET_KEY:
        return None
    
    return boto3.client(
        's3',
        endpoint_url=POLYGON_S3_ENDPOINT,
        aws_access_key_id=POLYGON_S3_ACCESS_KEY,
        aws_secret_access_key=POLYGON_S3_SECRET_KEY
    )

def list_available_dates():
    """List available dates in Flat Files."""
    s3 = get_s3_client()
    if not s3:
        return []
    
    try:
        # List dates in us_options_opra/minute_aggs_v1/
        prefix = "us_options_opra/minute_aggs_v1/"
        response = s3.list_objects_v2(Bucket=POLYGON_S3_BUCKET, Prefix=prefix, Delimiter='/')
        
        dates = []
        for prefix_info in response.get('CommonPrefixes', []):
            date_str = prefix_info['Prefix'].replace(prefix, '').replace('/', '')
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date_str)
            except:
                continue
        
        return sorted(dates)
    except Exception as e:
        print(f"Error listing dates: {e}")
        return []

def aggregate_and_save_daily_options_volume(date_str):
    """
    Pre-aggregate all options volume by ticker for a date and save to CSV.
    Creates options-day-aggs-[date].csv with ticker and total_daily_volume columns.
    """
    os.makedirs(AGGREGATED_CACHE_DIR, exist_ok=True)
    
    aggregated_file = os.path.join(AGGREGATED_CACHE_DIR, f"options-day-aggs-{date_str}.csv")
    
    # Check if already aggregated
    if os.path.exists(aggregated_file):
        print(f"    Using existing aggregated file: {aggregated_file}")
        return aggregated_file
    
    print(f"    Aggregating options volume for {date_str}...")
    
    # Download the raw flat file
    df = download_options_min_aggs(date_str)
    if df is None or df.empty:
        print(f"    [ERROR] Could not load flat file for {date_str}")
        print(f"    This could mean:")
        print(f"      1. Flat file doesn't exist for this date (check Massive API availability)")
        print(f"      2. S3 credentials are incorrect")
        print(f"      3. Date format is wrong (expected YYYY-MM-DD)")
        return None
    
    print(f"    [OK] Loaded flat file: {len(df):,} rows, {len(df.columns)} columns")
    
    # Extract underlying ticker and option type from options ticker format O:TICKERYYMMDDC/PSTRIKE
    def extract_ticker_info(opt_ticker):
        """Extract underlying ticker and option type from O:AAPL240426C00150000 format."""
        try:
            if not str(opt_ticker).startswith('O:'):
                return None, None
            ticker_part = str(opt_ticker).replace('O:', '')
            # Find first digit (start of date)
            for i, char in enumerate(ticker_part):
                if char.isdigit():
                    underlying = ticker_part[:i]
                    # Extract option type (C or P) - it's before the strike
                    # Format: TICKER + YYMMDD + C/P + STRIKE
                    if len(ticker_part) >= i + 7:
                        option_type_char = ticker_part[i + 6]  # 6 digits for date, then C or P
                        option_type = 'CALL' if option_type_char == 'C' else 'PUT'
                    else:
                        option_type = None
                    return underlying.upper() if underlying else None, option_type
            return None, None
        except:
            return None, None
    
    # Extract underlying tickers and option types
    ticker_info = df['ticker'].apply(extract_ticker_info)
    df['underlying_ticker'] = ticker_info.apply(lambda x: x[0] if x else None)
    df['option_type'] = ticker_info.apply(lambda x: x[1] if x else None)
    
    # Filter out rows where we couldn't extract ticker
    original_count = len(df)
    df_before_filter = df.copy()  # Keep copy for diagnostics
    df = df[df['underlying_ticker'].notna()].copy()
    
    if df.empty:
        print(f"    [ERROR] Could not extract underlying tickers from {original_count:,} rows")
        print(f"    This suggests the ticker format in flat files is different than expected")
        print(f"    Expected format: O:TICKERYYMMDDC/PSTRIKE (e.g., O:AAPL240426C00150000)")
        if original_count > 0 and 'ticker' in df_before_filter.columns:
            print(f"    Sample ticker values from flat file: {df_before_filter['ticker'].head(5).tolist()}")
        return None
    
    print(f"    [OK] Extracted {df['underlying_ticker'].nunique():,} unique underlying tickers from {len(df):,} contracts")
    
    # Calculate bid/ask side using Lee-Ready Algorithm (Tick Rule) + Price Position Analysis
    # Lee-Ready: Uses price change direction (close vs open) to infer trade direction
    # This is the quant standard method for determining ask/bid side from aggregate data
    def calculate_bid_ask_side(row):
        """
        Calculate bid/ask side using Lee-Ready Algorithm (Tick Rule) + Price Position Analysis.
        
        Lee-Ready Algorithm:
        - If close > open → price went up → likely ask side (buyer-initiated)
        - If close < open → price went down → likely bid side (seller-initiated)
        
        Combined with price position in high-low range for better accuracy.
        """
        contract_open = row.get('open', 0) or 0
        contract_close = row.get('close', 0) or 0
        contract_high = row.get('high', 0) or 0
        contract_low = row.get('low', 0) or 0
        volume = row.get('volume', 0) or 0
        
        if contract_close <= 0 or contract_open <= 0:
            return 0.5, 0.5  # Default 50/50 if no price data
        
        # METHOD 1: Lee-Ready Tick Rule (Primary Signal - 60% weight)
        # Price change direction is the strongest indicator
        price_change = contract_close - contract_open
        price_change_pct = price_change / contract_open if contract_open > 0 else 0
        
        # METHOD 2: Price Position in Range (Secondary Signal - 30% weight)
        # Where does close sit within the high-low range?
        price_range = contract_high - contract_low
        if price_range > 0:
            position_in_range = (contract_close - contract_low) / price_range
            # 0.0 = at low (bid side), 1.0 = at high (ask side)
        else:
            position_in_range = 0.5
        
        # METHOD 3: Volume-Weighted Confidence (10% weight)
        # Higher volume trades are more reliable indicators
        volume_confidence = min(1.0, volume / 100)  # Normalize to 0-1
        
        # Apply Lee-Ready Tick Rule
        if price_change_pct > 0.001:  # Price went up (buyer-initiated)
            tick_rule_ask = 0.7 + min(abs(price_change_pct) * 100, 0.3)  # 0.7 to 1.0
            tick_rule_bid = 1.0 - tick_rule_ask
        elif price_change_pct < -0.001:  # Price went down (seller-initiated)
            tick_rule_bid = 0.7 + min(abs(price_change_pct) * 100, 0.3)  # 0.7 to 1.0
            tick_rule_ask = 1.0 - tick_rule_bid
        else:  # No significant change (use position in range)
            tick_rule_ask = 0.5
            tick_rule_bid = 0.5
        
        # Position in range component
        position_ask = position_in_range  # Higher position = more ask side
        position_bid = 1.0 - position_in_range
        
        # Combine signals with weights
        ask_side_pct = (
            0.60 * tick_rule_ask +      # Primary: Lee-Ready tick rule
            0.30 * position_ask +      # Secondary: Position in range
            0.10 * (0.5 + 0.5 * volume_confidence)  # Volume confidence (slight bias for high volume)
        )
        bid_side_pct = 1.0 - ask_side_pct
        
        # Clamp to reasonable range (avoid extreme values)
        ask_side_pct = max(0.2, min(0.8, ask_side_pct))
        bid_side_pct = 1.0 - ask_side_pct
        
        return ask_side_pct, bid_side_pct
    
    # Calculate bid/ask side for each row
    bid_ask = df.apply(calculate_bid_ask_side, axis=1)
    df['ask_side_pct'] = bid_ask.apply(lambda x: x[0])
    df['bid_side_pct'] = bid_ask.apply(lambda x: x[1])
    
    # Calculate flow contributions
    # Bullish flow = CALLs at ask + PUTs at bid
    # Bearish flow = PUTs at ask + CALLs at bid
    df['bullish_flow'] = 0.0
    df['bearish_flow'] = 0.0
    
    # For CALLs: ask side = bullish, bid side = bearish
    call_mask = df['option_type'] == 'CALL'
    df.loc[call_mask, 'bullish_flow'] = df.loc[call_mask, 'volume'] * df.loc[call_mask, 'ask_side_pct']
    df.loc[call_mask, 'bearish_flow'] = df.loc[call_mask, 'volume'] * df.loc[call_mask, 'bid_side_pct']
    
    # For PUTs: bid side = bullish, ask side = bearish
    put_mask = df['option_type'] == 'PUT'
    df.loc[put_mask, 'bullish_flow'] = df.loc[put_mask, 'volume'] * df.loc[put_mask, 'bid_side_pct']
    df.loc[put_mask, 'bearish_flow'] = df.loc[put_mask, 'volume'] * df.loc[put_mask, 'ask_side_pct']
    
    # Aggregate by underlying ticker
    aggregated = df.groupby('underlying_ticker').agg({
        'volume': 'sum',  # Total volume
        'bullish_flow': 'sum',  # Sum of bullish flow
        'bearish_flow': 'sum',  # Sum of bearish flow
    }).reset_index()
    
    # Rename underlying_ticker to ticker
    aggregated = aggregated.rename(columns={'underlying_ticker': 'ticker'})
    
    # Ensure correct data types
    aggregated['volume'] = aggregated['volume'].astype(int)
    aggregated['bullish_flow'] = aggregated['bullish_flow'].astype(float)
    aggregated['bearish_flow'] = aggregated['bearish_flow'].astype(float)
    
    # Sort by volume descending
    aggregated = aggregated.sort_values('volume', ascending=False)
    
    # Select only the 4 required columns: ticker, volume, bullish_flow, bearish_flow
    aggregated = aggregated[['ticker', 'volume', 'bullish_flow', 'bearish_flow']]
    
    # Save to CSV
    aggregated.to_csv(aggregated_file, index=False)
    
    print(f"    [OK] Aggregated {len(aggregated):,} tickers, saved to {aggregated_file}")
    if len(aggregated) > 0:
        top_5 = aggregated.head(5)[['ticker', 'volume', 'bullish_flow', 'bearish_flow']].to_dict('records')
        print(f"      Top 5 by volume:")
        for item in top_5:
            print(f"        {item['ticker']}: {item['volume']:,} contracts (bullish: {item['bullish_flow']:,.0f}, bearish: {item['bearish_flow']:,.0f})")
        print(f"      Volume range: {aggregated['volume'].min():,} to {aggregated['volume'].max():,} contracts")
    else:
        print(f"    [WARNING] Aggregation produced 0 tickers - check if flat file has valid options data")
    
    return aggregated_file

def download_options_min_aggs(date_str):
    """
    Download options minute aggregates CSV for a specific date.
    Returns pandas DataFrame or None if not available.
    """
    s3 = get_s3_client()
    if not s3:
        print("S3 credentials not configured. Set POLYGON_S3_ACCESS_KEY and POLYGON_S3_SECRET_KEY in .env")
        return None
    
    # Check local cache first
    cache_file = os.path.join(FLAT_FILES_CACHE_DIR, f"options-min-aggs-{date_str}.csv")
    if os.path.exists(cache_file):
        print(f"Using cached file: {cache_file}")
        try:
            df = pd.read_csv(cache_file)
            # DEBUG: Check volume column in cached file
            volume_cols = [col for col in df.columns if 'volume' in col.lower() or col.lower() == 'vol']
            if not volume_cols:
                print(f"    [WARNING] No volume column in cached file! Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Error reading cached file: {e}")
    
    # Download from S3 (path format: us_options_opra/minute_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz)
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    year = date_obj.strftime('%Y')
    month = date_obj.strftime('%m')
    s3_key = f"us_options_opra/minute_aggs_v1/{year}/{month}/{date_str}.csv.gz"
    
    try:
        print(f"Downloading {s3_key} from Massive API Flat Files...")
        response = s3.get_object(Bucket=POLYGON_S3_BUCKET, Key=s3_key)
        
        # Read CSV (may be gzipped)
        content = response['Body'].read()
        
        # Try to decompress if gzipped
        try:
            content = gzip.decompress(content)
        except:
            pass  # Not gzipped
        
        # Parse CSV
        df = pd.read_csv(BytesIO(content))
        
        # DEBUG: Check if volume column exists and what it's called
        volume_cols = [col for col in df.columns if 'volume' in col.lower() or col.lower() == 'vol']
        if not volume_cols:
            print(f"    [WARNING] No volume column found in CSV! Columns: {list(df.columns)}")
            print(f"    First row sample: {df.iloc[0].to_dict() if not df.empty else 'Empty'}")
        else:
            print(f"    [OK] Volume column found: {volume_cols[0]}")
            if not df.empty:
                sample_vol = df[volume_cols[0]].iloc[0]
                print(f"    Sample volume value: {sample_vol} (type: {type(sample_vol)})")
        
        # Cache locally
        os.makedirs(FLAT_FILES_CACHE_DIR, exist_ok=True)
        df.to_csv(cache_file, index=False)
        print(f"Cached to {cache_file}")
        
        return df
    except s3.exceptions.NoSuchKey:
        print(f"File not found: {s3_key} (may not be available yet)")
        return None
    except Exception as e:
        print(f"Error downloading {s3_key}: {e}")
        return None

def filter_options_data(df, ticker, date_str):
    """
    Filter options data for a specific underlying ticker and date.
    
    Args:
        df: DataFrame from Flat Files CSV
        ticker: Underlying ticker (e.g., 'AAPL')
        date_str: Date string (YYYY-MM-DD)
    
    Returns:
        Filtered DataFrame with contract data
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Ensure ticker is uppercase and stripped
    ticker = str(ticker).strip().upper()
    
    # Options tickers in Flat Files are in format: O:TICKERYYMMDDC/PSTRIKE
    # Filter for contracts matching the underlying ticker
    ticker_prefix = f"O:{ticker}"
    
    # Ensure ticker column is string type and handle any whitespace
    if 'ticker' not in df.columns:
        return pd.DataFrame()
    
    # Convert to string and strip whitespace
    df['ticker'] = df['ticker'].astype(str).str.strip()
    
    # Filter for contracts starting with O:TICKER
    filtered = df[df['ticker'].str.startswith(ticker_prefix, na=False)].copy()
    
    # Debug: Show if filtering found anything (first time only per ticker/date)
    debug_key = f"{ticker}_{date_str}"
    if debug_key not in getattr(filter_options_data, '_debug_shown', set()):
        if filtered.empty:
            # Check if any tickers start with O: at all
            o_tickers = df[df['ticker'].str.startswith('O:', na=False)]
            if not o_tickers.empty:
                # Show sample of what we're looking for vs what exists
                sample_existing = o_tickers['ticker'].head(5).tolist()
                print(f"    [DEBUG] Looking for '{ticker_prefix}*' but found {len(o_tickers)} O:* tickers")
                print(f"    [DEBUG] Sample existing: {sample_existing}")
        else:
            print(f"    [DEBUG] Found {len(filtered)} rows for {ticker} (prefix: {ticker_prefix})")
        
        if not hasattr(filter_options_data, '_debug_shown'):
            filter_options_data._debug_shown = set()
        filter_options_data._debug_shown.add(debug_key)
    
    # Parse options ticker to extract contract details
    def parse_options_ticker(opt_ticker):
        """Parse O:AAPL241220C00150000 format."""
        try:
            # Remove O: prefix
            contract_str = opt_ticker.replace('O:', '')
            
            # Format: TICKER + YYMMDD + C/P + STRIKE (8 digits)
            # Work backwards: last 8 digits are strike, then C/P, then 6 digits for date
            if len(contract_str) < 15:  # Minimum: 1 char ticker + 6 date + 1 type + 8 strike
                return None
            
            # Strike is last 8 characters
            strike_str = contract_str[-8:]
            # Type is 9th from end
            type_char = contract_str[-9]
            # Date is 15th to 10th from end (6 digits)
            expiry_str = contract_str[-15:-9]
            # Ticker is everything before the date
            ticker_part = contract_str[:-15]
            
            # Convert to readable format
            expiry_date = datetime.strptime(expiry_str, '%y%m%d').date()
            strike = float(strike_str) / 1000
            option_type = 'CALL' if type_char == 'C' else 'PUT'
            
            return {
                'expiry': expiry_date,
                'strike': strike,
                'option_type': option_type
            }
        except Exception as e:
            return None
    
    # Add parsed contract details
    parsed = filtered['ticker'].apply(parse_options_ticker)
    filtered['contract_details'] = parsed
    
    # Debug: Check parsing success rate
    if debug_key not in getattr(filter_options_data, '_parse_debug_shown', set()):
        total_rows = len(filtered)
        parsed_success = filtered['contract_details'].notna().sum()
        if total_rows > 0 and parsed_success < total_rows:
            failed_count = total_rows - parsed_success
            sample_failed = filtered[filtered['contract_details'].isna()]['ticker'].head(3).tolist()
            print(f"    [DEBUG] {ticker}: Parsed {parsed_success}/{total_rows} contracts successfully")
            if sample_failed:
                print(f"      Sample failed tickers: {sample_failed}")
                # Show what the format looks like
                sample_ticker = sample_failed[0]
                print(f"      Sample format: '{sample_ticker}' (length: {len(sample_ticker)})")
        if not hasattr(filter_options_data, '_parse_debug_shown'):
            filter_options_data._parse_debug_shown = set()
        filter_options_data._parse_debug_shown.add(debug_key)
    
    # Filter out rows where parsing failed
    filtered = filtered[filtered['contract_details'].notna()]
    
    # Expand contract details into columns
    if not filtered.empty:
        filtered['expiry'] = filtered['contract_details'].apply(lambda x: x['expiry'] if x else None)
        filtered['strike'] = filtered['contract_details'].apply(lambda x: x['strike'] if x else None)
        filtered['option_type'] = filtered['contract_details'].apply(lambda x: x['option_type'] if x else None)
        filtered = filtered.drop('contract_details', axis=1)
    else:
        # Debug: Show why parsing failed
        if debug_key not in getattr(filter_options_data, '_empty_debug_shown', set()):
            original_filtered = df[df['ticker'].str.startswith(ticker_prefix, na=False)]
            if not original_filtered.empty:
                sample_tickers = original_filtered['ticker'].head(5).tolist()
                print(f"    [DEBUG] {ticker}: All {len(original_filtered)} contracts failed parsing")
                print(f"      Sample ticker formats: {sample_tickers}")
                for sample in sample_tickers[:2]:
                    print(f"        '{sample}' (length: {len(sample)})")
            if not hasattr(filter_options_data, '_empty_debug_shown'):
                filter_options_data._empty_debug_shown = set()
            filter_options_data._empty_debug_shown.add(debug_key)
    
    return filtered

def aggregate_daily_from_minutes(df):
    """
    Aggregate minute-level data to daily aggregates.
    Returns DataFrame with daily OHLCV per contract.
    Preserves contract details (expiry, strike, option_type) which are constant per ticker.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Build aggregation dictionary
    agg_dict = {
        'volume': 'sum',
        'open': 'first',  # First open of the day
        'close': 'last',   # Last close of the day
        'high': 'max',
        'low': 'min',
    }
    
    # Add transactions if it exists
    if 'transactions' in df.columns:
        agg_dict['transactions'] = 'sum'
    
    # Preserve contract details (expiry, strike, option_type) - these are constant per ticker
    if 'expiry' in df.columns:
        agg_dict['expiry'] = 'first'  # Should be same for all rows with same ticker
    if 'strike' in df.columns:
        agg_dict['strike'] = 'first'  # Should be same for all rows with same ticker
    if 'option_type' in df.columns:
        agg_dict['option_type'] = 'first'  # Should be same for all rows with same ticker
    
    # Group by ticker and aggregate
    daily = df.groupby('ticker').agg(agg_dict).reset_index()
    
    return daily

# Global flag to print CSV structure once per date
_csv_structure_checked = {}

def get_ticker_volume_from_aggregated(date_str, ticker):
    """
    Get total daily options volume and flow data for a ticker from pre-aggregated CSV.
    Returns dict with total_volume, bullish_flow, bearish_flow or None if not found.
    Note: CSV now only contains ticker, volume, bullish_flow, bearish_flow columns.
    """
    aggregated_file = os.path.join(AGGREGATED_CACHE_DIR, f"options-day-aggs-{date_str}.csv")
    
    if not os.path.exists(aggregated_file):
        # Try to create it
        aggregate_and_save_daily_options_volume(date_str)
    
    if not os.path.exists(aggregated_file):
        return None
    
    try:
        df = pd.read_csv(aggregated_file)
        ticker_upper = str(ticker).strip().upper()
        result = df[df['ticker'] == ticker_upper]
        
        if result.empty:
            return None
        
        row = result.iloc[0]
        return {
            'total_volume': int(row['volume']),
            'bullish_flow': float(row.get('bullish_flow', 0)),
            'bearish_flow': float(row.get('bearish_flow', 0))
        }
    except Exception as e:
        print(f"    [ERROR] Failed to read aggregated file: {e}")
        return None

def select_top_tickers_from_aggregated(date_str, num_tickers, min_volume_threshold=500, balance_flow=True):
    """
    Read aggregated CSV file and select top N tickers by volume.
    If balance_flow=True, randomly selects tickers with net flow >= 1 std dev from 0.
    This reduces circular logic by not selecting absolute extremes.
    
    Selection logic:
    1. Take top 100 tickers by volume
    2. Calculate mean and std of net flow scores
    3. Filter for tickers where |net_flow| >= 1 std dev from mean
    4. Randomly sample 20 bullish + 20 bearish from qualified pools
    
    Returns list of ticker strings.
    """
    aggregated_file = os.path.join(AGGREGATED_CACHE_DIR, f"options-day-aggs-{date_str}.csv")
    
    if not os.path.exists(aggregated_file):
        # Try to create it
        aggregate_and_save_daily_options_volume(date_str)
    
    if not os.path.exists(aggregated_file):
        print(f"    [ERROR] Aggregated file not found: {aggregated_file}")
        return []
    
    try:
        df = pd.read_csv(aggregated_file)
        
        # Diagnostic: Show volume distribution before filtering
        if len(df) > 0:
            max_vol = df['volume'].max()
            median_vol = df['volume'].median()
            tickers_above_threshold = len(df[df['volume'] >= min_volume_threshold])
            print(f"    Volume stats: max={max_vol:,.0f}, median={median_vol:,.0f}, above threshold={tickers_above_threshold}")
        
        # Filter by minimum volume threshold
        df = df[df['volume'] >= min_volume_threshold].copy()
        
        if df.empty:
            print(f"    [WARNING] No tickers meet volume threshold of {min_volume_threshold:,}")
            print(f"    Consider lowering MIN_VOLUME_THRESHOLD or checking if flat file has data for {date_str}")
            return []
        
        # Calculate net flow score using bullish/bearish flow
        # Net flow = bullish_flow - bearish_flow (raw difference, already incorporates volume via ask/bid logic)
        df['net_flow_score'] = df['bullish_flow'] - df['bearish_flow']
        df['net_flow_score'] = df['net_flow_score'].fillna(0)
        
        # Sort by volume descending
        df = df.sort_values('volume', ascending=False)
        
        if not balance_flow:
            # Simple: take top N by volume
            selected = df.head(num_tickers)
        else:
            # Statistical approach: Randomly select from tickers with |net_flow| >= 1 std dev from 0
            # This reduces circular logic by not selecting absolute extremes
            import numpy as np
            import random
            
            top_100 = df.head(min(100, len(df))).copy()
            
            if len(top_100) < 2:
                # Fallback: not enough data, just take top N
                selected = top_100.head(num_tickers)
            else:
                # Calculate mean and std of net flow scores
                net_flow_mean = top_100['net_flow_score'].mean()
                net_flow_std = top_100['net_flow_score'].std()
                
                # Define number of bullish/bearish tickers needed (must be before they're used)
                num_bullish = num_tickers // 2
                num_bearish = num_tickers - num_bullish
                
                # Handle edge case: if std is 0 or very small, use absolute threshold
                if net_flow_std < 1e-6:
                    # All net flows are similar, use absolute threshold (top/bottom 25%)
                    threshold = top_100['net_flow_score'].abs().quantile(0.75)
                    bullish_pool = top_100[top_100['net_flow_score'] >= threshold].copy()
                    bearish_pool = top_100[top_100['net_flow_score'] <= -threshold].copy()
                else:
                    # Use 0.15 standard deviation threshold (less restrictive to ensure enough tickers)
                    # This ensures we get enough tickers while still filtering for meaningful flow
                    threshold = net_flow_std * 0.15
                    bullish_pool = top_100[top_100['net_flow_score'] >= (net_flow_mean + threshold)].copy()
                    bearish_pool = top_100[top_100['net_flow_score'] <= (net_flow_mean - threshold)].copy()
                    
                    # If still not enough, use even lower threshold (0.1 std dev as fallback)
                    if len(bullish_pool) < num_bullish or len(bearish_pool) < num_bearish:
                        threshold = net_flow_std * 0.1
                        bullish_pool = top_100[top_100['net_flow_score'] >= (net_flow_mean + threshold)].copy()
                        bearish_pool = top_100[top_100['net_flow_score'] <= (net_flow_mean - threshold)].copy()
                
                # Randomly sample from pools (using date as seed for reproducibility)
                # Convert date string to integer seed for consistency
                date_seed = hash(date_str) % (2**32)
                random.seed(date_seed)
                np.random.seed(date_seed)
                
                # Randomly sample from pools
                selected_bullish = []
                selected_bearish = []
                
                if len(bullish_pool) >= num_bullish:
                    selected_bullish = bullish_pool.sample(n=min(num_bullish, len(bullish_pool)), random_state=date_seed).copy()
                elif len(bullish_pool) > 0:
                    # Not enough bullish tickers, take all available
                    selected_bullish = bullish_pool.copy()
                    print(f"      [WARNING] Only {len(bullish_pool)} bullish tickers meet threshold (requested {num_bullish})")
                
                if len(bearish_pool) >= num_bearish:
                    selected_bearish = bearish_pool.sample(n=min(num_bearish, len(bearish_pool)), random_state=date_seed).copy()
                elif len(bearish_pool) > 0:
                    # Not enough bearish tickers, take all available
                    selected_bearish = bearish_pool.copy()
                    print(f"      [WARNING] Only {len(bearish_pool)} bearish tickers meet threshold (requested {num_bearish})")
                
                # Combine selections
                if len(selected_bullish) > 0 and len(selected_bearish) > 0:
                    selected = pd.concat([selected_bullish, selected_bearish])
                elif len(selected_bullish) > 0:
                    selected = selected_bullish
                elif len(selected_bearish) > 0:
                    selected = selected_bearish
                else:
                    # Fallback: no tickers meet threshold, use top/bottom by net flow
                    print(f"      [WARNING] No tickers meet 1 std dev threshold, falling back to top/bottom selection")
                    top_100_sorted = top_100.sort_values('net_flow_score', ascending=False)
                    selected = pd.concat([
                        top_100_sorted.head(num_bullish),
                        top_100_sorted.tail(num_bearish)
                    ])
        
        tickers = selected['ticker'].str.upper().str.strip().tolist()
        
        print(f"    Selected {len(tickers)} tickers from aggregated CSV")
        if balance_flow and len(selected) > 0:
            if 'net_flow_score' in selected.columns:
                bullish_tickers = selected[selected['net_flow_score'] > 0]
                bearish_tickers = selected[selected['net_flow_score'] < 0]
                if len(bullish_tickers) > 0:
                    max_bullish = bullish_tickers.loc[bullish_tickers['net_flow_score'].idxmax()]
                    print(f"      Bullish sample: {len(bullish_tickers)} tickers (max: {max_bullish['ticker']}, net flow: {max_bullish['net_flow_score']:+.2f})")
                if len(bearish_tickers) > 0:
                    min_bearish = bearish_tickers.loc[bearish_tickers['net_flow_score'].idxmin()]
                    print(f"      Bearish sample: {len(bearish_tickers)} tickers (min: {min_bearish['ticker']}, net flow: {min_bearish['net_flow_score']:+.2f})")
        
        return tickers
        
    except Exception as e:
        print(f"    [ERROR] Failed to read aggregated file: {e}")
        import traceback
        traceback.print_exc()
        return []

def fetch_options_data_from_flat_files(ticker, date_str):
    """
    Fetch options data for a ticker/date from Flat Files instead of API calls.
    Returns data in same format as fetch_historical_options_data().
    """
    global _csv_structure_checked
    
    # Download CSV for the date
    df = download_options_min_aggs(date_str)
    if df is None or df.empty:
        return None
    
    # DEBUG: Check CSV structure once per date (not per ticker)
    if date_str not in _csv_structure_checked:
        print(f"\n  [DEBUG] Checking flat files structure for {date_str}...")
        print(f"    CSV columns: {list(df.columns)}")
        print(f"    CSV shape: {df.shape}")
        if not df.empty:
            print(f"    Sample row columns: {list(df.iloc[0].index)}")
            sample_row = df.iloc[0].to_dict()
            print(f"    Sample values: {list(sample_row.keys())[:10]}...")
            # Check for volume column (case-insensitive)
            volume_cols = [col for col in df.columns if 'volume' in col.lower() or 'vol' in col.lower()]
            if volume_cols:
                print(f"    [OK] Found volume column(s): {volume_cols}")
                print(f"    Sample volume value: {df[volume_cols[0]].iloc[0] if len(volume_cols) > 0 else 'N/A'}")
            else:
                print(f"    [WARNING] NO VOLUME COLUMN FOUND! Available columns: {list(df.columns)}")
        _csv_structure_checked[date_str] = True
    
    # Filter for this ticker
    ticker_data = filter_options_data(df, ticker, date_str)
    if ticker_data.empty:
        # Debug: Check if ticker prefix exists at all in the file
        if date_str not in _csv_structure_checked or _csv_structure_checked.get(date_str + '_ticker_check') is None:
            # Check what tickers actually exist in the file
            if 'ticker' in df.columns:
                sample_tickers = df['ticker'].head(1000).unique()
                underlying_tickers = set()
                for t in sample_tickers:
                    if str(t).startswith('O:'):
                        # Extract underlying ticker (everything after O: until first digit)
                        ticker_part = str(t).replace('O:', '')
                        # Find first digit
                        for i, char in enumerate(ticker_part):
                            if char.isdigit():
                                underlying = ticker_part[:i]
                                if underlying:
                                    underlying_tickers.add(underlying)
                                break
                if underlying_tickers:
                    print(f"    [DEBUG] Sample underlying tickers in file: {sorted(list(underlying_tickers))[:20]}")
            _csv_structure_checked[date_str + '_ticker_check'] = True
        return None
    
    # Aggregate to daily (if minute-level data)
    # NOTE: Each row in daily_data represents ONE options contract
    # The volume column is the volume for THAT SPECIFIC CONTRACT
    if 'window_start' in ticker_data.columns:
        daily_data = aggregate_daily_from_minutes(ticker_data)
    else:
        daily_data = ticker_data
    
    # Convert to flow format
    # IMPORTANT: We will sum volume across ALL contracts (rows) to get total options volume per stock
    flows = []
    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
    
    # Check for volume column (try multiple possible names)
    volume_col = None
    for col in daily_data.columns:
        if col.lower() in ['volume', 'vol', 'v']:
            volume_col = col
            break
    
    if volume_col is None:
        # DEBUG: Report missing volume column
        if date_str not in _csv_structure_checked or _csv_structure_checked.get(date_str + '_vol_warned') is None:
            print(f"    [WARNING] No volume column found in daily data for {ticker} on {date_str}")
            print(f"    Available columns: {list(daily_data.columns)}")
            _csv_structure_checked[date_str + '_vol_warned'] = True
        return None  # Can't process without volume
    
    # Iterate through each contract (row) and collect volume
    # Each row = one options contract, volume = volume for that contract
    total_contracts = len(daily_data)
    total_volume_sum = 0
    
    # Debug: Check if required columns exist
    if date_str not in _csv_structure_checked or _csv_structure_checked.get(date_str + '_columns_checked') is None:
        required_cols = ['expiry', 'strike', 'option_type']
        missing_cols = [col for col in required_cols if col not in daily_data.columns]
        if missing_cols:
            print(f"    [DEBUG] {ticker}: Missing columns in daily_data: {missing_cols}")
            print(f"      Available columns: {list(daily_data.columns)}")
            if not daily_data.empty:
                print(f"      Sample row keys: {list(daily_data.iloc[0].index)}")
        _csv_structure_checked[date_str + '_columns_checked'] = True
    
    skipped_no_volume = 0
    skipped_no_close = 0
    skipped_no_expiry_strike = 0
    
    for _, row in daily_data.iterrows():
        volume = row.get(volume_col, 0)
        # Handle NaN or None
        if pd.isna(volume) or volume is None:
            volume = 0
        
        if volume < 1:
            skipped_no_volume += 1
            continue
        
        contract_close = row.get('close', 0)
        if contract_close <= 0:
            skipped_no_close += 1
            continue
        
        # Parse contract details
        expiry = row.get('expiry')
        strike = row.get('strike')
        option_type = row.get('option_type', 'CALL')
        
        if not expiry or not strike:
            skipped_no_expiry_strike += 1
            continue
        
        premium = contract_close * volume * 100
        contract_high = row.get('high', contract_close * 1.02) or contract_close * 1.02
        contract_low = row.get('low', contract_close * 0.98) or contract_close * 0.98
        mid = (contract_high + contract_low) / 2 if (contract_high + contract_low) > 0 else contract_close
        
        # Ask/bid side determination using Lee-Ready Algorithm (Tick Rule)
        # Lee-Ready: Uses price change direction (close vs open) to infer trade direction
        contract_open = row.get('open', contract_close * 0.99) or contract_close * 0.99
        
        if contract_open <= 0:
            contract_open = contract_close * 0.99
        
        # Lee-Ready Tick Rule: Price change direction
        price_change = contract_close - contract_open
        price_change_pct = price_change / contract_open if contract_open > 0 else 0
        
        # Price position in range (secondary signal)
        price_range = contract_high - contract_low
        if price_range > 0:
            position_in_range = (contract_close - contract_low) / price_range
        else:
            position_in_range = 0.5
        
        # Volume confidence
        volume_confidence = min(1.0, volume / 100)
        
        # Apply Lee-Ready Algorithm
        if price_change_pct > 0.001:  # Price went up → ask side (buyer-initiated)
            tick_rule_ask = 0.7 + min(abs(price_change_pct) * 100, 0.3)
            tick_rule_bid = 1.0 - tick_rule_ask
        elif price_change_pct < -0.001:  # Price went down → bid side (seller-initiated)
            tick_rule_bid = 0.7 + min(abs(price_change_pct) * 100, 0.3)
            tick_rule_ask = 1.0 - tick_rule_bid
        else:  # No significant change
            tick_rule_ask = 0.5
            tick_rule_bid = 0.5
        
        # Combine signals
        ask_side_pct = (
            0.60 * tick_rule_ask +
            0.30 * position_in_range +
            0.10 * (0.5 + 0.5 * volume_confidence)
        )
        bid_side_pct = 1.0 - ask_side_pct
        
        # Clamp to reasonable range
        ask_side_pct = max(0.2, min(0.8, ask_side_pct))
        bid_side_pct = 1.0 - ask_side_pct
        
        dte = (expiry - date_obj).days
        if dte >= 0:
            flows.append({
                'option_type': option_type,
                'premium': premium,
                'ask_side_pct': ask_side_pct,
                'bid_side_pct': bid_side_pct,
                'dte': dte,
                'volume': volume  # Volume for THIS contract
            })
            total_volume_sum += volume
    
    # Debug: Show why flows might be empty
    if not flows:
        debug_key = f"{ticker}_{date_str}"
        if date_str not in _csv_structure_checked or _csv_structure_checked.get(debug_key + '_empty_flows') is None:
            print(f"    [DEBUG] {ticker}: No flows created from {total_contracts} contracts")
            print(f"      Skipped - no volume: {skipped_no_volume}")
            print(f"      Skipped - no close price: {skipped_no_close}")
            print(f"      Skipped - no expiry/strike: {skipped_no_expiry_strike}")
            if total_contracts > 0 and skipped_no_expiry_strike == total_contracts:
                print(f"      [WARNING] All contracts missing expiry/strike columns!")
                print(f"      Available columns: {list(daily_data.columns)}")
                if not daily_data.empty:
                    sample_row = daily_data.iloc[0]
                    print(f"      Sample row expiry: {sample_row.get('expiry', 'MISSING')}")
                    print(f"      Sample row strike: {sample_row.get('strike', 'MISSING')}")
            _csv_structure_checked[debug_key + '_empty_flows'] = True
        return None
    
    # DEBUG: Show that we're summing across contracts
    # The total volume per stock is the sum of volume across all contracts (rows)
    if date_str not in _csv_structure_checked or _csv_structure_checked.get(date_str + '_vol_sum_shown') is None:
        call_vol = sum(f['volume'] for f in flows if f['option_type'] == 'CALL')
        put_vol = sum(f['volume'] for f in flows if f['option_type'] == 'PUT')
        print(f"    [DEBUG] {ticker}: {total_contracts} contracts, Total volume: {total_volume_sum:,} contracts")
        print(f"      CALL volume: {call_vol:,}, PUT volume: {put_vol:,}")
        _csv_structure_checked[date_str + '_vol_sum_shown'] = True
    
    # Still need ticker details for market cap (1 API call per ticker, not per contract)
    # This is much better than 100+ calls per ticker!
    # Could also cache ticker details or get from a separate file
    market_cap = 0
    stock_price = None
    close_price = daily_data['close'].iloc[0] if not daily_data.empty else 0
    
    # Try to get market cap from API (only 1 call per ticker, not per contract)
    # NOTE: If this fails, we still return the data - market cap is not required for volume calculation
    try:
        from massive import RESTClient
        API_KEY = os.getenv("MASSIVE_API_KEY")
        if API_KEY:
            massive_client = RESTClient(API_KEY)
            details = massive_client.get_ticker_details(ticker)
            market_cap = getattr(details, 'market_cap', 0)
            shares_outstanding = getattr(details, 'weighted_shares_outstanding', 0) or \
                               getattr(details, 'share_class_shares_outstanding', 0)
            stock_price = (market_cap / shares_outstanding) if shares_outstanding and market_cap else None
    except Exception as e:
        # Market cap API call failed - this is OK, we can still use the data
        # Only log first few failures to avoid spam
        if date_str not in _csv_structure_checked or _csv_structure_checked.get(date_str + '_api_warned') is None:
            print(f"    [WARN] Could not get market cap for {ticker} (API call failed), continuing anyway")
            _csv_structure_checked[date_str + '_api_warned'] = True
        pass  # Market cap not critical, can use defaults
    
    return {
        'ticker': ticker,
        'date': date_str,
        'market_cap': market_cap,
        'stock_price': stock_price,
        'close_price': close_price,
        'flows': flows
    }

def check_flat_files_access():
    """Check if Flat Files access is configured and working."""
    s3 = get_s3_client()
    if not s3:
        return False, "S3 credentials not configured"
    
    try:
        # Try to list bucket
        s3.list_objects_v2(Bucket=POLYGON_S3_BUCKET, MaxKeys=1)
        return True, "Flat Files access configured and working"
    except Exception as e:
        return False, f"Error accessing Flat Files: {e}"

def validate_flat_files_have_volume(date_str):
    """
    Validate that flat files contain OPTIONS CONTRACT volume data (not stock volume).
    Raises an error if volume column is missing.
    Returns True if volume exists, raises ValueError if not.
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING FLAT FILES STRUCTURE")
    print(f"{'='*80}")
    print(f"Checking date: {date_str}")
    print(f"Expected: OPTIONS CONTRACT volume (not stock volume)")
    
    # Download or load the CSV
    df = download_options_min_aggs(date_str)
    if df is None or df.empty:
        raise ValueError(f"ERROR: Could not load flat file for {date_str}. File may not exist or is empty.")
    
    print(f"[OK] File loaded successfully")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    
    # Verify this is options data (ticker should start with O:)
    if 'ticker' in df.columns and not df.empty:
        sample_tickers = df['ticker'].head(5).tolist()
        options_tickers = [t for t in sample_tickers if str(t).startswith('O:')]
        if options_tickers:
            print(f"[OK] Confirmed: Options contract data (ticker format: O:TICKER...)")
            print(f"  Sample tickers: {options_tickers[:3]}")
        else:
            print(f"[WARNING] Tickers don't start with 'O:' prefix!")
            print(f"  Sample tickers: {sample_tickers[:3]}")
            print(f"  This might be stock data, not options data!")
    
    # Check for volume column (case-insensitive)
    volume_cols = [col for col in df.columns if 'volume' in col.lower() or col.lower() == 'vol']
    
    if not volume_cols:
        print(f"\n{'='*80}")
        print(f"[ERROR] VOLUME COLUMN NOT FOUND IN FLAT FILES!")
        print(f"{'='*80}")
        print(f"Available columns: {list(df.columns)}")
        print(f"\nThe flat files do not contain volume data.")
        print(f"This makes backtesting unfeasible as OPTIONS VOLUME is required to:")
        print(f"  - Filter tickers by options contract volume")
        print(f"  - Calculate flow scores")
        print(f"  - Determine bid/ask side percentages")
        print(f"\nPlease check:")
        print(f"  1. Are you using the correct flat files format?")
        print(f"  2. Do the files need to be processed differently?")
        print(f"  3. Is volume data in a different column name?")
        print(f"\nSample row data:")
        if not df.empty:
            sample = df.iloc[0].to_dict()
            for key, value in list(sample.items())[:10]:
                print(f"  {key}: {value}")
        raise ValueError("Flat files do not contain volume data. Backtesting cannot proceed.")
    
    volume_col = volume_cols[0]
    print(f"[OK] Volume column found: '{volume_col}'")
    
    # Check if volume data actually exists (not all zeros/NaN)
    if volume_col in df.columns:
        non_zero_volumes = df[df[volume_col].notna() & (df[volume_col] > 0)]
        if non_zero_volumes.empty:
            print(f"\n{'='*80}")
            print(f"[ERROR] VOLUME COLUMN EXISTS BUT ALL VALUES ARE ZERO/NaN!")
            print(f"{'='*80}")
            print(f"Volume column '{volume_col}' exists but contains no valid data.")
            print(f"Total rows: {len(df):,}")
            print(f"Rows with volume > 0: {len(non_zero_volumes):,}")
            raise ValueError("Volume column exists but contains no valid data. Backtesting cannot proceed.")
        
        # Check if volume values make sense for OPTIONS CONTRACTS
        # Options volume should typically be in hundreds/thousands of contracts
        # Stock volume would be in millions/billions of shares
        sample_volumes = df[volume_col].head(100).tolist()
        max_sample_vol = max(sample_volumes) if sample_volumes else 0
        median_vol = df[volume_col].median()
        
        print(f"[OK] Volume data validated")
        print(f"  Rows with volume > 0: {len(non_zero_volumes):,} / {len(df):,}")
        print(f"  Sample volume values: {df[volume_col].head(5).tolist()}")
        print(f"  Volume range: {df[volume_col].min():.0f} to {df[volume_col].max():.0f}")
        print(f"  Median volume: {median_vol:.0f}")
        
        # Warn if volumes look like stock volume (millions+) instead of options volume
        if max_sample_vol > 1_000_000:
            print(f"\n  [WARNING] Volume values are very high (max: {max_sample_vol:,.0f})")
            print(f"     Options contract volume is typically in hundreds/thousands.")
            print(f"     Stock volume is typically in millions/billions.")
            print(f"     This might be STOCK VOLUME, not OPTIONS VOLUME!")
            print(f"     Please verify the flat files contain OPTIONS contract data.")
        else:
            print(f"  [OK] Volume values look reasonable for OPTIONS CONTRACTS")
    
    print(f"{'='*80}\n")
    return True

if __name__ == "__main__":
    # Test Flat Files access
    print("Testing Flat Files access...")
    has_access, message = check_flat_files_access()
    print(f"Access: {has_access}")
    print(f"Message: {message}")
    
    if has_access:
        # Test download with a known good date
        test_date = "2024-12-20"  # Use a date we know exists
        print(f"\nTesting download for {test_date}...")
        df = download_options_min_aggs(test_date)
        if df is not None:
            print(f"[OK] Downloaded {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            print(f"\nSample data:")
            print(df.head(3))
            
            # Test filtering for a specific ticker
            print(f"\nTesting filter for AAPL...")
            aapl_data = filter_options_data(df, 'AAPL', test_date)
            print(f"[OK] Found {len(aapl_data)} AAPL contracts")
            if not aapl_data.empty:
                print(aapl_data.head(3))
        else:
            print("[FAIL] Download failed or file not available")
