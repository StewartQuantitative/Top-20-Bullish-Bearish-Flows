"""
Filter CBOE weekly optionable stocks by open interest threshold
Downloads the CBOE list and filters to stocks with significant options activity
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except:
        pass

# Load environment and initialize Massive API
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)
API_KEY = os.getenv("MASSIVE_API_KEY")
if not API_KEY:
    print(f"ERROR: MASSIVE_API_KEY not found in .env file!\n  Looked for .env at: {env_path}")
    exit(1)

from massive import RESTClient
massive_client = RESTClient(API_KEY)

# Configuration
CBOE_CSV_FILENAME = "weeklysmf.csv"
CBOE_CSV_URL = "https://www.cboe.com/us/options/symboldir/weeklys_options/?download=csv"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_OPEN_INTEREST = 150000
MAX_WORKERS = 50

def download_and_extract_cboe_tickers():
    """Download CBOE CSV and extract tickers."""
    csv_path = os.path.join(OUTPUT_DIR, CBOE_CSV_FILENAME)
    
    # Download if needed (cache valid for 12 hours)
    should_download = True
    if os.path.exists(csv_path):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(csv_path))
        if datetime.now() - file_mod_time < timedelta(hours=12):
            print(f"Using cached CBOE CSV (downloaded at {file_mod_time.strftime('%Y-%m-%d %H:%M')}, cache valid for 12 hours)")
            should_download = False
    
    if should_download:
        print("Downloading CBOE CSV...")
        with open(csv_path, 'wb') as f:
            f.write(requests.get(CBOE_CSV_URL, timeout=30).content)
    
    # Extract tickers from CSV
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # Find ticker column
        ticker_col = 'Stock Symbol' if 'Stock Symbol' in df.columns else df.columns[1]
        tickers = df[ticker_col].dropna().str.strip().str.upper()
        
        # Filter valid tickers (1-5 chars, alphanumeric with dots/dashes)
        valid_tickers = [t for t in tickers if 1 <= len(t) <= 5 and t.replace('.', '').replace('-', '').isalnum()]
        return sorted(set(valid_tickers))
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return []

def get_ticker_open_interest(ticker):
    """Get total open interest for a ticker."""
    try:
        contracts = list(massive_client.list_snapshot_options_chain(ticker))
        return sum(int(getattr(c, 'open_interest', 0) or 0) for c in contracts)
    except:
        return 0

def filter_by_open_interest(tickers, min_oi=MIN_OPEN_INTEREST):
    """Filter tickers by minimum open interest using parallel processing."""
    print(f"\nChecking open interest for {len(tickers)} tickers...")
    print(f"Minimum open interest threshold: {min_oi:,}")
    print(f"Using {MAX_WORKERS} parallel workers\n")
    
    filtered_tickers = []
    ticker_oi_map = {}
    processed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_ticker_open_interest, ticker): ticker for ticker in tickers}
        
        for future in as_completed(futures):
            processed += 1
            if processed % 50 == 0:
                print(f"  Processed {processed}/{len(tickers)} tickers...")
            
            try:
                ticker = futures[future]
                oi = future.result()
                ticker_oi_map[ticker] = oi
                if oi >= min_oi:
                    filtered_tickers.append(ticker)
            except Exception as e:
                print(f"  Error processing {futures[future]}: {e}")
    
    print(f"\nCompleted processing {len(tickers)} tickers")
    print(f"Found {len(filtered_tickers)} tickers with open interest >= {min_oi:,}\n")
    return filtered_tickers, ticker_oi_map

def save_results(filtered_tickers, ticker_oi_map):
    """Save filtered list to CSV file."""
    filtered_path = os.path.join(OUTPUT_DIR, "stocks_with_significant_options_flow.csv")
    
    if filtered_tickers:
        filtered_df = pd.DataFrame([
            {'Ticker': t, 'Total_Open_Interest': ticker_oi_map[t]} 
            for t in filtered_tickers
        ]).sort_values('Total_Open_Interest', ascending=False)
        filtered_df.to_csv(filtered_path, index=False)
        print(f"[OK] Filtered list saved to: {filtered_path}")
        print(f"     Filtered tickers: {len(filtered_tickers)}")
        print(f"     Average open interest: {filtered_df['Total_Open_Interest'].mean():,.0f}")
        print(f"     Median open interest: {filtered_df['Total_Open_Interest'].median():,.0f}")
    else:
        pd.DataFrame(columns=['Ticker', 'Total_Open_Interest']).to_csv(filtered_path, index=False)
        print(f"[WARNING] Filtered list saved to: {filtered_path}")
        print(f"     No tickers found with open interest >= {MIN_OPEN_INTEREST:,}")
    
    return filtered_path

def main():
    print("="*80)
    print("CBOE Weekly Optionable Stocks - Open Interest Filter")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("Step 1: Downloading CBOE weekly optionable stocks list...")
    all_tickers = download_and_extract_cboe_tickers()
    if not all_tickers:
        print("ERROR: No tickers found in CBOE CSV!")
        return
    print(f"[OK] Loaded {len(all_tickers)} tickers from CBOE list\n")
    
    print("Step 2: Filtering by open interest...")
    filtered_tickers, ticker_oi_map = filter_by_open_interest(all_tickers, MIN_OPEN_INTEREST)
    
    print("Step 3: Saving results...")
    filtered_path = save_results(filtered_tickers, ticker_oi_map)
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nFile saved to: {OUTPUT_DIR}")
    print(f"  - Filtered list: {os.path.basename(filtered_path)}")

if __name__ == "__main__":
    main()
