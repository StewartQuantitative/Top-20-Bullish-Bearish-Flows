# Options Flow Scoring System

A comprehensive system for analyzing options flow data to identify bullish and bearish signals, with backtesting capabilities to validate and optimize the scoring algorithm.

## Overview

This system consists of two main components:

1. **Flow Strength Program** (`top_bullish_bearish_flows.py`) - Real-time options flow analysis
2. **Backtesting System** (`Backtesting/`) - Historical validation and parameter optimization

## Features

- **Real-time Options Flow Analysis**: Identifies stocks with significant bullish/bearish options flow
- **Market Cap Categorization**: Separates stocks by market cap (mega, large, mid, small)
- **ETF Analysis**: Tracks options flow for major ETFs
- **Historical Backtesting**: Validates flow scoring algorithm on historical data
- **Parameter Optimization**: Uses Bayesian optimization to find optimal scoring parameters
- **Comprehensive Reporting**: Generates detailed reports with visualizations

## Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Massive API Key** - Sign up at [massive.com](https://massive.com)
3. **Required packages**: See `requirements.txt` (if provided) or install:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy python-dotenv optuna massive
   ```

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Top-bullish-bearish-flows
   ```

2. **Create `.env` file** in the root directory:
   ```env
   MASSIVE_API_KEY=your_actual_api_key_here
   ```
   
   **Note**: Replace `your_actual_api_key_here` with your actual Massive API key. Never commit this file to version control.

3. **Run the flow strength program**:
   ```bash
   python top_bullish_bearish_flows.py
   ```

4. **Run quick backtest** (optional, for validation):
   ```bash
   cd Backtesting
   python quick_backtest.py
   ```

## Project Structure

```
Top-bullish-bearish-flows/
├── top_bullish_bearish_flows.py    # Main flow strength program
├── CSV_Output/                      # Output CSV files
├── Backtesting/                    # Backtesting system
│   ├── backtest_flow_correlation.py  # Full backtest script
│   ├── quick_backtest.py            # Quick validation test
│   ├── flat_files_loader.py          # Historical data loader
│   └── md/                           # Documentation
│       ├── README.md                 # Backtesting overview
│       ├── GETTING_STARTED.md        # Quick start guide
│       └── OVERVIEW.md               # System overview
└── README.md                        # This file
```

## Main Program: Flow Strength Analysis

The `top_bullish_bearish_flows.py` script:

- Fetches real-time options chain data from Massive API
- Calculates bullish/bearish flow scores using configurable parameters
- Categorizes stocks by market cap
- Exports results to CSV files

### Configuration

Edit the configuration section in `top_bullish_bearish_flows.py`:

```python
# Performance Settings
MAX_CONTRACTS_PER_TICKER = 300
MAX_WORKERS = 100

# Output Settings
TOP_N_STOCKS = 20
TOP_N_ETFS = 10
TOP_N_PER_CATEGORY = 10
```

## Backtesting System

The backtesting system validates the flow scoring algorithm on historical data and optimizes parameters.

### Quick Backtest (5-10 minutes)

```bash
cd Backtesting
python quick_backtest.py
```

Tests a small sample of dates to validate the system works.

### Full Backtest (2-4 hours)

```bash
cd Backtesting
python backtest_flow_correlation.py
```

Comprehensive backtest with:
- 100 random trading dates
- 60 tickers per date
- Bayesian parameter optimization (50 trials)
- Detailed reports and visualizations

### Output

Backtest results are saved to:
- `backtest_results/` - Full backtest output
- `quick_backtest_results/` - Quick test output

Includes:
- Correlation analysis (IC, t-statistics, hit rates)
- Parameter optimization results
- Visualizations (scatter plots, heatmaps, time series)
- Detailed markdown reports

## Flow Scoring Algorithm

The flow score combines:

1. **Bullish Flow**: Premium-weighted options bought at ask (aggressive buying)
2. **Bearish Flow**: Premium-weighted options sold at bid (aggressive selling)
3. **DTE Decay**: Longer-dated options weighted less
4. **Market Cap Adjustment**: Normalizes for company size
5. **Volatility Normalization**: Optional volatility-based adjustment

### Key Parameters

- `ASK_WEIGHT`: Weight for aggressive trades (default: 1.0)
- `BID_WEIGHT`: Weight for passive trades (default: 0.7)
- `DTE_LAMBDA`: Decay factor for longer-dated options (default: 0.10)

These parameters are optimized during backtesting.

## Documentation

- **Main Documentation**: See `Backtesting/md/README.md`
- **Getting Started**: See `Backtesting/md/GETTING_STARTED.md`
- **System Overview**: See `Backtesting/md/OVERVIEW.md`

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- python-dotenv
- optuna
- massive (Massive API client)

## Security

**Important**: Never commit your API key to the repository. The system uses environment variables loaded from a `.env` file, which is excluded via `.gitignore`.

## License

[Add your license here]

## Contributing

[Add contribution guidelines if applicable]

## Support

[Add support information if applicable]
