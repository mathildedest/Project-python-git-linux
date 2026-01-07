import requests
import pandas as pd
import numpy as np


def fetch_price_history(asset_id: str, vs_currency: str = "eur", days: int = 365) -> pd.DataFrame:
    """
    Récupère l'historique de prix d'un actif via l'API CoinGecko.

    Parameters
    ----------
    asset_id : str
        Identifiant CoinGecko ("bitcoin", "ethereum", etc.)
    vs_currency : str
        Devise de cotation ("eur" par défaut).
    days : int
        Nombre de jours d'historique à récupérer.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{asset_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}

    # Ici je reste volontairement simple : un seul appel, pas de logique de retry avancée
    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        # Dans l'app Streamlit je gère ça avec un message d'erreur plutôt que de crasher
        raise ValueError(f"API error for {asset_id}: {response.status_code}")

    data = response.json()

    # CoinGecko renvoie une liste de [timestamp_ms, price]
    prices = data.get("prices")
    if not prices:
        raise ValueError(f"No 'prices' field in API response for {asset_id}")

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


def build_price_matrix(assets: dict, vs_currency: str = "eur", days: int = 365) -> pd.DataFrame:
    """
    Récupère les prix pour plusieurs actifs et les met dans un seul DataFrame.

    Parameters
    ----------
    assets : dict
        mapping label -> coingecko_id (ex: {"Bitcoin (BTC)": "bitcoin"}).
    """
    all_series = []
    for label, asset_id in assets.items():
        df_asset = fetch_price_history(asset_id, vs_currency=vs_currency, days=days)
        # Je renomme la colonne price avec le label de l'actif pour être plus lisible
        df_asset = df_asset.rename(columns={"price": label})
        all_series.append(df_asset)

    if not all_series:
        raise ValueError("No assets provided")

    # Concaténation sur l'index temps, puis on enlève les lignes avec NaN
    prices = pd.concat(all_series, axis=1).dropna()
    return prices


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rendements simples : (P_t / P_{t-1} - 1).
    """
    returns = price_df.pct_change().dropna()
    return returns


def compute_portfolio_returns(returns: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Rendements du portefeuille à partir des rendements individuels et de poids.

    Hypothèses simplificatrices :
    - poids statiques (pas de rebalancement effectif dans ce code)
    - pas de coûts de transaction
    """
    # On aligne l'ordre des poids avec les colonnes du DataFrame
    w = np.array([weights.get(col, 0.0) for col in returns.columns], dtype=float)
    if w.sum() == 0:
        # Pour éviter division par zéro, je laisse l'app gérer ce cas
        raise ValueError("Sum of weights is zero.")

    # Normalisation au cas où la somme n'est pas exactement 1
    w = w / w.sum()

    port_ret = (returns * w).sum(axis=1)
    return port_ret


def basic_stats(returns: pd.Series, periods_per_year: int = 365) -> dict:
    """
    Quelques stats basiques sur le portefeuille.

    - rendement moyen
    - volatilité
    - version annualisée
    - drawdown max
    - valeur finale (en partant de 1)
    """
    if returns.empty:
        return {}

    mean_ret = returns.mean()
    vol = returns.std()

    # Annualisation vraiment simple (comme en cours)
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
