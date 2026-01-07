import requests
import pandas as pd
import numpy as np


def fetch_price_history(asset_id: str, vs_currency: str = "eur", days: int = 365) -> pd.DataFrame:
    """
    Fetch historical prices for a single asset from the CoinGecko API.

    Parameters
    ----------
    asset_id : str
        CoinGecko ID (e.g. "bitcoin", "ethereum").
    vs_currency : str
        Quote currency ("eur" by default).
    days : int
        Number of days of history to fetch.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{asset_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}

    # Simple call, no retry logic to keep things readable
    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        # In the Streamlit app, I catch this and display an error instead of crashing
        raise ValueError(f"API error for {asset_id}: {response.status_code}")

    data = response.json()

    # CoinGecko returns a list of [timestamp_ms, price]
    prices = data.get("prices")
    if not prices:
        raise ValueError(f"No 'prices' field in API response for {asset_id}")

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


def build_price_matrix(assets: dict, vs_currency: str = "eur", days: int = 365) -> pd.DataFrame:
    """
    Fetch prices for several assets and build a single price matrix.

    Parameters
    ----------
    assets : dict
        Mapping label -> CoinGecko ID (e.g. {"Bitcoin (BTC)": "bitcoin"}).
    """
    all_series = []
    for label, asset_id in assets.items():
        df_asset = fetch_price_history(asset_id, vs_currency=vs_currency, days=days)
        # Rename 'price' column with asset label to keep the matrix readable
        df_asset = df_asset.rename(columns={"price": label})
        all_series.append(df_asset)

    if not all_series:
        raise ValueError("No assets provided")

    # Concatenate on the time index and drop rows with missing values
    prices = pd.concat(all_series, axis=1).dropna()
    return prices


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple returns: (P_t / P_{t-1} - 1).
    """
    returns = price_df.pct_change().dropna()
    return returns


def compute_portfolio_returns(returns: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Compute portfolio returns from individual asset returns and weights.

    Simplifying assumptions:
    - static weights (no actual rebalancing in this code),
    - no transaction costs.
    """
    # Align weights with the order of the columns in the returns DataFrame
    w = np.array([weights.get(col, 0.0) for col in returns.columns], dtype=float)
    if w.sum() == 0:
        # I let the app handle this case to avoid division by zero
        raise ValueError("Sum of weights is zero.")

    # Normalize in case the sum is not exactly 1
    w = w / w.sum()

    port_ret = (returns * w).sum(axis=1)
    return port_ret


def basic_stats(returns: pd.Series, periods_per_year: int = 365) -> dict:
    """
    Basic portfolio statistics.

    - mean return
    - volatility
    - annualized versions
    - max drawdown
    - final value (starting from 1)
    """
    if returns.empty:
        return {}

    mean_ret = returns.mean()
    vol = returns.std()

    # Very simple annualization (as in lectures)
    ann_ret = (1 + mean_ret) ** periods_per_year - 1
    ann_vol = vol * np.sqrt(periods_per_year)

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_dd = drawdown.min()

    stats = {
        "mean_return": mean_ret,
        "volatility": vol,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "max_drawdown": max_dd,
        "final_value": cumulative.iloc[-1],
    }
    return stats
