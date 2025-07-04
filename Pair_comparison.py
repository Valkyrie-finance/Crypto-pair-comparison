# Symbol definitions - Edit these to analyze different pairs
SYMBOL1 = "RRX"  # First symbol to analyze (Gold futures)
SYMBOL2 = "XLM"  # Second symbol to analyze
SYMBOL = "XRP"   # Single symbol for beta analysis

# Analysis period in days
DAYS = 30  # Default number of days for analysis

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from typing import List, Dict
from functools import lru_cache
import yfinance as yf

class RateLimiter:
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def wait_if_needed(self):
        now = time.time()
        # Remove calls older than the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            # Wait until the oldest call is outside the time window
            sleep_time = self.calls[0] + self.time_window - now
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.calls.append(now)

# Create rate limiter for Binance API
binance_rate_limiter = RateLimiter(max_calls=1200, time_window=60)

@lru_cache(maxsize=1000)
def get_historical_prices(symbol: str, days: int = 30) -> List[float]:
    """
    Get historical price data for a symbol.
    Tries Binance → Hyperliquid → Yahoo Finance in order.
    Returns a list of closing prices.
    """
    for fetcher in [fetch_binance_prices, fetch_hyperliquid_prices, fetch_yahoo_prices]:
        try:
            prices = fetcher(symbol, days)
            if prices:
                return prices
        except Exception:
            continue
    
    return []

def fetch_binance_prices(symbol: str, days: int) -> List[float]:
    binance_rate_limiter.wait_if_needed()

    if not symbol.endswith("USDT"):
        symbol += "USDT"

    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
        "endTime": int(datetime.now().timestamp() * 1000),
        "limit": days
    }
    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    prices = [float(candle[4]) for candle in data]

    # Check if we have enough data
    if len(prices) < days:
        raise ValueError(f"Insufficient price history for {symbol} on Binance ({len(prices)} days available)")

    return ensure_data_integrity(prices, days, "Binance", symbol)

def fetch_hyperliquid_prices(symbol: str, days: int) -> List[float]:
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    url = "https://api.hyperliquid.xyz/info"
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol.upper(),
            "interval": "1d",
            "startTime": start_time,
            "endTime": end_time
        }
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data)
    prices = df['c'].astype(float).tolist()

    return ensure_data_integrity(prices, days, "Hyperliquid", symbol)

def fetch_yahoo_prices(symbol: str, days: int) -> List[float]:
    df = yf.Ticker(symbol).history(period=f"{days}d", interval="1d")

    if df.empty:
        raise ValueError("No data returned from Yahoo Finance")

    prices = df['Close'].tolist()
    return ensure_data_integrity(prices, days, "Yahoo Finance", symbol)

def ensure_data_integrity(prices: List[float], days: int, source: str, symbol: str) -> List[float]:
    if len(prices) > days:
        return prices[-days:]
    elif len(prices) < days:
        print(f"Insufficient price history for {symbol} on {source} ({len(prices)} days available)")
        raise ValueError("Insufficient data")
    return prices

def calculate_beta(symbol_prices: list, btc_prices: list) -> float:
    """
    Calculate beta value of a coin relative to BTC using log returns.
    Returns beta value, defaults to 1.0 on failure.
    """
    try:
        if not validate_prices(symbol_prices, btc_prices):
            return 1.0

        # Special case: BTC vs BTC → beta = 1
        if symbol_prices == btc_prices:
            return 1.0

        symbol_returns = compute_log_returns(symbol_prices)
        btc_returns = compute_log_returns(btc_prices)

        if not validate_returns(symbol_returns, btc_returns):
            return 1.0

        beta = compute_beta(symbol_returns, btc_returns)

        if abs(beta) > 10 or np.isnan(beta):
            print(f"Warning: Unusually high or invalid beta value: {beta}")
            return 1.0

        return round(beta, 3)

    except Exception as e:
        print(f"Error calculating beta: {e}")
        return 1.0

def validate_prices(symbol_prices: list, btc_prices: list) -> bool:
    if len(symbol_prices) != len(btc_prices):
        print("Warning: Price arrays have different lengths")
        return False
    return True

def compute_log_returns(prices: list) -> np.ndarray:
    return np.array([
        np.log(prices[i] / prices[i - 1])
        for i in range(1, len(prices))
    ])

def validate_returns(symbol_returns: np.ndarray, btc_returns: np.ndarray) -> bool:
    if len(symbol_returns) == 0 or len(btc_returns) == 0:
        print("Warning: No returns data available")
        return False
    if np.all(symbol_returns == 0) or np.all(btc_returns == 0):
        print("Warning: All returns are zero")
        return False
    return True

def compute_beta(symbol_returns: np.ndarray, btc_returns: np.ndarray) -> float:
    covariance = np.cov(symbol_returns, btc_returns, ddof=0)[0][1]
    btc_variance = np.var(btc_returns, ddof=0)

    if btc_variance == 0 or np.isnan(covariance) or np.isnan(btc_variance):
        print("Warning: Invalid variance or covariance")
        return 1.0

    return covariance / btc_variance

def get_coin_beta(symbol: str, days: int = DAYS) -> float:
    """
    Compute beta of a given coin relative to BTC.
    Falls back to 1.0 if data is unavailable or an error occurs.
    """
    try:
        btc_prices = get_historical_prices("BTCUSDT", days)
        symbol_prices = get_historical_prices(symbol, days)

        if not btc_prices or not symbol_prices:
            print(f"Warning: Missing price data for {symbol} or BTCUSDT")
            return 1.0

        return calculate_beta(symbol_prices, btc_prices)

    except Exception as e:
        print(f"Error getting beta for {symbol}: {e}")
        return 1.0

def calculate_correlation(symbol1: str, symbol2: str, days: int = DAYS) -> float:
    """
    Calculate Pearson correlation coefficient between two assets based on daily returns.
    Returns 0.0 if data is missing or error occurs.
    """
    try:
        prices1 = get_historical_prices(symbol1, days)
        prices2 = get_historical_prices(symbol2, days)

        if not prices1 or not prices2:
            print(f"Error: Could not get price data for {symbol1} or {symbol2}")
            return 0.0

        # Align to shortest available length
        min_len = min(len(prices1), len(prices2))
        prices1, prices2 = prices1[-min_len:], prices2[-min_len:]

        # Compute daily returns
        returns1 = np.diff(prices1) / prices1[:-1]
        returns2 = np.diff(prices2) / prices2[:-1]

        # Check if we have meaningful returns
        if returns1.size == 0 or returns2.size == 0:
            print("Warning: Insufficient return data")
            return 0.0

        # Calculate correlation
        correlation = np.corrcoef(returns1, returns2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return 0.0

def analyze_pair_btc_relationship(symbol1: str, symbol2: str, days: int = DAYS) -> Dict:
    """
    Analyze relationship between two symbols based on BTC movements
    Returns dictionary with analysis results
    """
    try:
        def safe_get_prices(symbol):
            prices = get_historical_prices(symbol, days)
            return prices if prices and len(prices) >= days else None

        btc_prices = safe_get_prices("BTCUSDT")
        prices1 = safe_get_prices(symbol1)
        prices2 = safe_get_prices(symbol2)

        if not btc_prices or not prices1 or not prices2:
            print(f"Error: Missing or invalid price data")
            return {}

        if any(price <= 0 for price in btc_prices + prices1 + prices2):
            print("Error: Found invalid (zero or negative) prices in data")
            return {}

        def daily_returns(prices):
            return np.diff(prices) / prices[:-1]

        btc_returns = daily_returns(btc_prices)
        returns1 = daily_returns(prices1)
        returns2 = daily_returns(prices2)

        if not len(btc_returns) or not len(returns1) or not len(returns2):
            print("Error: Could not calculate returns")
            return {}

        if np.all(btc_returns == 0) or np.all(returns1 == 0) or np.all(returns2 == 0):
            print("Error: All returns are zero")
            return {}

        beta1 = calculate_beta(prices1, btc_prices)
        beta2 = calculate_beta(prices2, btc_prices)
        hedge_ratio = beta1 / beta2 if beta2 != 0 else 1.0

        pair_returns = returns1 - returns2
        beta_neutral_returns = returns1 - (hedge_ratio * returns2)

        btc_up_days = btc_returns > 0
        btc_down_days = btc_returns < 0

        def strategy_metrics(returns, mask):
            prob = np.mean(returns[mask] > 0) * 100
            avg = np.mean(returns[mask]) * 100
            return prob, avg

        up_day_prob, up_day_avg = strategy_metrics(pair_returns, btc_up_days)
        down_day_prob, down_day_avg = strategy_metrics(pair_returns, btc_down_days)

        beta_up_prob, beta_up_avg = strategy_metrics(beta_neutral_returns, btc_up_days)
        beta_down_prob, beta_down_avg = strategy_metrics(beta_neutral_returns, btc_down_days)

        def return_stats(returns):
            total_return = ((1 + returns).prod() - 1) * 100
            consistency = np.mean(returns > 0) * 100
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            win_rate = len(wins) / len(returns) * 100 if len(returns) > 0 else 0
            avg_win = np.mean(wins) * 100 if len(wins) > 0 else 0
            avg_loss = np.mean(losses) * 100 if len(losses) > 0 else 0
            avg_pnl = np.mean(returns) * 100
            return total_return, consistency, win_rate, avg_win, avg_loss, avg_pnl

        pair_metrics = return_stats(pair_returns)
        beta_metrics = return_stats(beta_neutral_returns)

        def sharpe(returns):
            std = np.std(returns) if len(returns) > 0 else 1
            vol = std * np.sqrt(DAYS)
            monthly_return = np.mean(returns) * DAYS
            return round(monthly_return / vol, 2) if vol > 0 else 0

        return {
            'btc_correlation1': np.corrcoef(returns1, btc_returns)[0, 1],
            'btc_correlation2': np.corrcoef(returns2, btc_returns)[0, 1],
            'pair_trade_correlation': np.corrcoef(pair_returns, btc_returns)[0, 1],
            'up_day_probability': up_day_prob,
            'down_day_probability': down_day_prob,
            'up_day_avg_return': up_day_avg,
            'down_day_avg_return': down_day_avg,
            'pair_trade_total_return': pair_metrics[0],
            'consistency': pair_metrics[1],
            'win_rate': pair_metrics[2],
            'avg_win': pair_metrics[3],
            'avg_loss': pair_metrics[4],
            'avg_pnl': pair_metrics[5],
            'sharpe_ratio': sharpe(pair_returns),
            'beta1': beta1,
            'beta2': beta2,
            'hedge_ratio': hedge_ratio,
            'beta_neutral_up_prob': beta_up_prob,
            'beta_neutral_down_prob': beta_down_prob,
            'beta_neutral_up_return': beta_up_avg,
            'beta_neutral_down_return': beta_down_avg,
            'beta_neutral_total_return': beta_metrics[0],
            'beta_neutral_consistency': beta_metrics[1],
            'beta_neutral_win_rate': beta_metrics[2],
            'beta_neutral_avg_win': beta_metrics[3],
            'beta_neutral_avg_loss': beta_metrics[4],
            'beta_neutral_avg_pnl': beta_metrics[5],
            'beta_neutral_sharpe_ratio': sharpe(beta_neutral_returns),
        }

    except Exception as e:
        print(f"Error in pair analysis: {e}")
        return {}

def calculate_position_sizes(symbol1: str, symbol2: str, days: int = DAYS, investment_amount: float = 10000) -> Dict:
    """
    Calculate position sizes for both delta neutral and beta neutral strategies.
    Returns dictionary with position sizes and expected returns.
    """
    def get_latest_price(prices: list) -> float:
        return prices[-1] if prices else 0.0

    try:
        # Get historical prices
        prices1 = get_historical_prices(symbol1, days)
        prices2 = get_historical_prices(symbol2, days)

        if not prices1 or not prices2:
            print("Error: Could not get price data for one or both symbols.")
            return {}

        # Current prices
        price1 = get_latest_price(prices1)
        price2 = get_latest_price(prices2)

        if price1 <= 0 or price2 <= 0:
            print("Error: Invalid (zero or negative) current prices.")
            return {}

        # Delta Neutral Strategy — equal capital in both
        half_investment = investment_amount / 2
        delta_neutral = {
            'symbol1_amount': half_investment,
            'symbol2_amount': half_investment,
            'symbol1_units': half_investment / price1,
            'symbol2_units': half_investment / price2
        }

        # Beta Neutral Strategy — weight investment to neutralize beta exposure
        beta = calculate_beta(prices2, prices1)  # beta of symbol2 relative to symbol1
        if beta <= 0:
            print("Warning: Non-positive beta, defaulting to delta neutral weights.")
            beta = 1.0

        symbol1_amount = investment_amount / (1 + beta)
        symbol2_amount = investment_amount - symbol1_amount

        beta_neutral = {
            'symbol1_amount': symbol1_amount,
            'symbol2_amount': symbol2_amount,
            'symbol1_units': symbol1_amount / price1,
            'symbol2_units': symbol2_amount / price2,
            'beta': round(beta, 3)
        }

        return {
            'delta_neutral': delta_neutral,
            'beta_neutral': beta_neutral,
            'current_prices': {
                symbol1: price1,
                symbol2: price2
            }
        }

    except Exception as e:
        print(f"Error calculating position sizes: {e}")
        return {}

def compare_strategies(symbol1: str, symbol2: str, days: int = DAYS, investment_amount: float = 10000) -> Dict:
    """
    Compare performance of delta neutral and beta neutral strategies.
    Returns dictionary with comparison results.
    """
    try:
        # Get position sizes
        positions = calculate_position_sizes(symbol1, symbol2, days, investment_amount)
        if not positions:
            return {}

        # Get historical prices
        prices1 = get_historical_prices(symbol1, days)
        prices2 = get_historical_prices(symbol2, days)

        if not prices1 or not prices2 or len(prices1) != len(prices2):
            print("Error: Invalid or mismatched price data.")
            return {}

        # Compute daily returns
        def compute_daily_returns(prices):
            return np.array([(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))])

        returns1 = compute_daily_returns(prices1)
        returns2 = compute_daily_returns(prices2)

        # Delta Neutral: long symbol1, short symbol2 (equal weights)
        delta_returns = returns1 - returns2

        # Beta Neutral: weighted based on beta
        beta = positions['beta_neutral'].get('beta', 1.0)
        if beta <= 0:
            print("Warning: Invalid beta, falling back to delta neutral weights.")
            beta = 1.0

        weight1 = 1 / (1 + beta)
        weight2 = beta / (1 + beta)
        beta_returns = (returns1 * weight1) - (returns2 * weight2)

        def summarize_strategy(returns: np.ndarray) -> Dict:
            total_return = (np.prod(1 + returns) - 1) * 100
            avg_return = np.mean(returns) * 100
            std_dev = np.std(returns) * 100
            sharpe = (np.mean(returns) / np.std(returns)) if np.std(returns) != 0 else 0
            return {
                'total_return': round(total_return, 2),
                'avg_return': round(avg_return, 4),
                'std_dev': round(std_dev, 4),
                'sharpe_ratio': round(sharpe, 2)
            }

        return {
            'positions': positions,
            'delta_neutral_stats': summarize_strategy(delta_returns),
            'beta_neutral_stats': summarize_strategy(beta_returns),
            'days_analyzed': days
        }

    except Exception as e:
        print(f"Error comparing strategies: {e}")
        return {}

def get_price_changes(symbol: str, days: int = DAYS) -> Dict:
    """
    Get percentage price change and start/end prices over a given number of days.
    """
    try:
        prices = get_historical_prices(symbol, days)
        if not prices or len(prices) < 2:
            print(f"Error: Not enough price data for {symbol}")
            return {}

        start_price = prices[0]
        end_price = prices[-1]
        total_change = ((end_price - start_price) / start_price) * 100

        return {
            'symbol': symbol,
            'start_price': round(start_price, 4),
            'end_price': round(end_price, 4),
            'total_change': round(total_change, 2)
        }

    except Exception as e:
        print(f"Error calculating price changes for {symbol}: {e}")
        return {}

def display_correlation_analysis(symbol1, symbol2, days, analysis):
    print(f"\nCorrelation with BTC:")
    print(f"{days}d {symbol1} correlation with BTC: {analysis['btc_correlation1']:.3f}")
    print(f"{days}d {symbol1} beta with BTC: {analysis['beta1']:.3f}")
    print(f"{days}d {symbol2} correlation with BTC: {analysis['btc_correlation2']:.3f}")
    print(f"{days}d {symbol2} beta with BTC: {analysis['beta2']:.3f}")
    print(f"{days}d Pair trade correlation with BTC (Long {symbol1}/Short {symbol2}): {analysis['pair_trade_correlation']:.3f}")
    print(f"{days}d Hedge ratio for beta neutrality: {analysis['hedge_ratio']:.3f}")

def display_strategy_performance(strategy_name, days, analysis):
    print(f"\n{strategy_name} Strategy:")
    print("\nPerformance on BTC Up Days:")
    if strategy_name == "Delta Neutral":
        print(f"{days}d Probability of profitable trade: {analysis['up_day_probability']:.1f}%")
        print(f"{days}d Average pair trade return: {analysis['up_day_avg_return']:.2f}%")
        
        print("\nPerformance on BTC Down Days:")
        print(f"{days}d Probability of profitable trade: {analysis['down_day_probability']:.1f}%")
        print(f"{days}d Average pair trade return: {analysis['down_day_avg_return']:.2f}%")
        
        print("\nTrading Consistency:")
        print(f"{days}d Consistency (profitable days): {analysis['consistency']:.1f}%")
        print(f"{days}d Win rate: {analysis['win_rate']:.1f}%")
        print(f"{days}d Average win: {analysis['avg_win']:.2f}%")
        print(f"{days}d Average loss: {analysis['avg_loss']:.2f}%")
        print(f"{days}d Average PnL: {analysis['avg_pnl']:.2f}%")
        print(f"{days}d Sharpe ratio: {analysis['sharpe_ratio']:.2f}")
    else:  # Beta Neutral
        print(f"{days}d Probability of profitable trade: {analysis['beta_neutral_up_prob']:.1f}%")
        print(f"{days}d Average pair trade return: {analysis['beta_neutral_up_return']:.2f}%")
        
        print("\nPerformance on BTC Down Days:")
        print(f"{days}d Probability of profitable trade: {analysis['beta_neutral_down_prob']:.1f}%")
        print(f"{days}d Average pair trade return: {analysis['beta_neutral_down_return']:.2f}%")
        
        print("\nTrading Consistency:")
        print(f"{days}d Consistency (profitable days): {analysis['beta_neutral_consistency']:.1f}%")
        print(f"{days}d Win rate: {analysis['beta_neutral_win_rate']:.1f}%")
        print(f"{days}d Average win: {analysis['beta_neutral_avg_win']:.2f}%")
        print(f"{days}d Average loss: {analysis['beta_neutral_avg_loss']:.2f}%")
        print(f"{days}d Average PnL: {analysis['beta_neutral_avg_pnl']:.2f}%")
        print(f"{days}d Sharpe ratio: {analysis['beta_neutral_sharpe_ratio']:.2f}")

def display_summary(symbol1, symbol2, days, btc_changes, symbol1_changes, symbol2_changes, analysis_7d, analysis_days):
    print("\nOverall Performance:")
    print(f"BTC change (7d/{days}d): {btc_changes['7d'].get('total_change', 0):.2f}% / {btc_changes[f'{days}d'].get('total_change', 0):.2f}%")
    print(f"{symbol1} change (7d/{days}d): {symbol1_changes['7d'].get('total_change', 0):.2f}% / {symbol1_changes[f'{days}d'].get('total_change', 0):.2f}%")
    print(f"{symbol2} change (7d/{days}d): {symbol2_changes['7d'].get('total_change', 0):.2f}% / {symbol2_changes[f'{days}d'].get('total_change', 0):.2f}%")
    print(f"Delta Neutral return (7d/{days}d): {analysis_7d['pair_trade_total_return']:.2f}% / {analysis_days['pair_trade_total_return']:.2f}%")
    print(f"Beta Neutral return (7d/{days}d): {analysis_7d['beta_neutral_total_return']:.2f}% / {analysis_days['beta_neutral_total_return']:.2f}%")

def display_recommendations(analysis):
    print("\nTrading Implications:")
    print("\nStrategy Comparison:")
    print("- Beta neutral strategy has shown better overall returns" if analysis['beta_neutral_total_return'] > analysis['pair_trade_total_return'] else "- Delta neutral strategy has shown better overall returns")
    print("- Beta neutral strategy has more consistent performance" if analysis['beta_neutral_consistency'] > analysis['consistency'] else "- Delta neutral strategy has more consistent performance")
    print("- Beta neutral strategy offers better risk-adjusted returns (higher Sharpe ratio)" if analysis['beta_neutral_sharpe_ratio'] > analysis['sharpe_ratio'] else "- Delta neutral strategy offers better risk-adjusted returns (higher Sharpe ratio)")

    print("\nStrategy Selection Guidelines:")
    print("- Significant beta difference between coins suggests beta neutral strategy may be more effective" if abs(analysis['beta1'] - analysis['beta2']) > 0.5 else "- Similar beta values suggest delta neutral strategy may be sufficient")
    print("- Large hedge ratio indicates beta neutral strategy requires significant position size adjustments" if analysis['hedge_ratio'] > 1.5 or analysis['hedge_ratio'] < 0.5 else "- Moderate hedge ratio suggests beta neutral strategy is practical to implement")

    print("\nRecommendation:")
    if all([
        analysis['beta_neutral_total_return'] > analysis['pair_trade_total_return'],
        analysis['beta_neutral_consistency'] > analysis['consistency'],
        analysis['beta_neutral_sharpe_ratio'] > analysis['sharpe_ratio']
    ]):
        print("- Beta neutral strategy is recommended based on superior performance across all metrics")
    elif all([
        analysis['pair_trade_total_return'] > analysis['beta_neutral_total_return'],
        analysis['consistency'] > analysis['beta_neutral_consistency'],
        analysis['sharpe_ratio'] > analysis['beta_neutral_sharpe_ratio']
    ]):
        print("- Delta neutral strategy is recommended based on superior performance across all metrics")
    else:
        print("- Consider using both strategies in combination or alternating based on market conditions")

def main():
    beta = get_coin_beta(SYMBOL, DAYS)
    print(f"\nBeta for {SYMBOL}: {beta:.2f}")

    btc_changes = {'7d': get_price_changes("BTCUSDT", 7), f'{DAYS}d': get_price_changes("BTCUSDT", DAYS)}
    symbol1_changes = {'7d': get_price_changes(SYMBOL1, 7), f'{DAYS}d': get_price_changes(SYMBOL1, DAYS)}
    symbol2_changes = {'7d': get_price_changes(SYMBOL2, 7), f'{DAYS}d': get_price_changes(SYMBOL2, DAYS)}

    print(f"\n=== Pair Relationship Analysis for {SYMBOL1} vs {SYMBOL2} ===")
    analysis_7d = analyze_pair_btc_relationship(SYMBOL1, SYMBOL2, days=7)
    analysis_days = analyze_pair_btc_relationship(SYMBOL1, SYMBOL2, days=DAYS)

    correlation = calculate_correlation(SYMBOL1, SYMBOL2, DAYS)
    print(f"\nCorrelation between {SYMBOL1} and {SYMBOL2}: {correlation:.3f}")

    if analysis_7d and analysis_days:
        display_correlation_analysis(SYMBOL1, SYMBOL2, DAYS, analysis_days)
        display_strategy_performance("Delta Neutral", DAYS, analysis_days)
        display_strategy_performance("Beta Neutral", DAYS, analysis_days)
        display_summary(SYMBOL1, SYMBOL2, DAYS, btc_changes, symbol1_changes, symbol2_changes, analysis_7d, analysis_days)
        display_recommendations(analysis_days)
    else:
        print("Analysis failed")

if __name__ == "__main__":
    main()