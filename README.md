# Crypto Pair Comparison Tool

This tool analyzes the performance, beta, and correlation between two crypto assets using historical price data from Binance, Hyperliquid, or Yahoo Finance. It also measures the consistency of its performance vs BTC price. 

## Features
- Calculates beta and correlation between two assets
- Analyzes pair trading strategies (delta neutral, beta neutral)
- Analyzes performance vs BTC price 
- Uses rolling historical price data
- Outputs performance metrics and recommendations

## Usage

1. Edit `Pair_comparison.py` to set your desired symbols and analysis period at the top of the file.
2. Run the script:
   ```bash
   python Pair_comparison.py
   ```
3. Review the output for strategy insights and recommendations.

## Requirements
See `requirements.txt` for dependencies. 
