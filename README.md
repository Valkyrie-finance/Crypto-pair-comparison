# Crypto Pair Comparison Tool

This tool analyzes and compares cryptocurrency pairs for trading strategies, including beta-neutral and delta-neutral approaches. It fetches historical price data from Binance, Hyperliquid, and Yahoo Finance, and provides statistical analysis and trading recommendations.

## Features
- Beta calculation relative to BTC
- Correlation analysis between crypto pairs
- Performance analysis of delta-neutral and beta-neutral strategies
- Historical price data from multiple sources
- Detailed trading recommendations

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Edit the top of `Pair_comparison.py` to set your desired symbols and analysis period.
3. Run the script:
   ```bash
   python Pair_comparison.py
   ```

## Example
Edit these lines at the top of the script:
```python
SYMBOL1 = "HYPE"  # First symbol to analyze
SYMBOL2 = "XLM"   # Second symbol to analyze
SYMBOL = "XRP"    # Single symbol for beta analysis
DAYS = 30          # Analysis period in days
```

## License
MIT License
