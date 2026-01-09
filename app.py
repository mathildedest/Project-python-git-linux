import os
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard")

DATA_PATH = "data/prices.csv"


# API
def get_price(coin_id: str, currency: str) -> float:
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coin_id, "vs_currencies": currency}

    r = requests.get(url, params=params, timeout=10)

    if r.status_code == 429:
        raise Exception("API limit reached. Please wait 1 minute.")

    r.raise_for_status()
    data = r.json()
    return float(data[coin_id][currency])


# Load data
def save_price(price: float) -> None:
    os.makedirs("data", exist_ok=True)
    now = pd.Timestamp.utcnow()

    new_row = pd.DataFrame([{"time": now, "price": price}])

    if os.path.exists(DATA_PATH):
        old = pd.read_csv(DATA_PATH)
        df = pd.concat([old, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(DATA_PATH, index=False)

@st.cache_data
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=["time", "price"])

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["time", "price"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

# Strategies
def buy_and_hold(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["price"].pct_change().fillna(0.0)
    out["strat_ret"] = out["ret"]
    out["strat_value"] = (1.0 + out["strat_ret"]).cumprod()
    return out

def sma_crossover(df: pd.DataFrame, fast: int = 5, slow: int = 10) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["price"].pct_change().fillna(0.0)

    out["sma_fast"] = out["price"].rolling(fast).mean()
    out["sma_slow"] = out["price"].rolling(slow).mean()

    out["pos"] = 0
    out.loc[out["sma_fast"] > out["sma_slow"], "pos"] = 1
    out["pos"] = out["pos"].fillna(0)

    # use previous position
    out["strat_ret"] = out["pos"].shift(1).fillna(0) * out["ret"]
    out["strat_value"] = (1.0 + out["strat_ret"]).cumprod()
    return out

def momentum_strategy(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["price"].pct_change().fillna(0.0)

    out["mom"] = out["price"].pct_change(window)
    out["pos"] = 0
    out.loc[out["mom"] > 0, "pos"] = 1

    out["strat_ret"] = out["pos"].shift(1).fillna(0) * out["ret"]
    out["strat_value"] = (1.0 + out["strat_ret"]).cumprod()
    return out

# Metrics
def max_drawdown(values: pd.Series) -> float:
    peak = values.cummax()
    dd = values / peak - 1.0
    return float(dd.min())


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 365) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    if r.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std())


def annual_volatility(returns: pd.Series, periods_per_year: int = 365) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.std())


def total_return(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    return float(values.iloc[-1] / values.iloc[0] - 1.0)

# Sidebar
st.sidebar.header("Settings")

coin = st.sidebar.selectbox("Asset", ["bitcoin", "ethereum"])
cur = st.sidebar.selectbox("Currency", ["eur", "usd"])

auto_refresh = st.sidebar.checkbox("Auto refresh (5 min)", value=False)
if auto_refresh:
    st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True)

st.sidebar.divider()

# anti-spam click
if "last_click" not in st.session_state:
    st.session_state.last_click = 0.0

if st.sidebar.button("Fetch & save price"):
    now = time.time()
    if now - st.session_state.last_click < 3:
        st.sidebar.warning("Wait 2-3 seconds and try again.")
    else:
        try:
            p = get_price(coin, cur)
            save_price(p)
            load_data.clear()  # refresh cache
            st.sidebar.success(f"Saved: {p:.2f} {cur.upper()}")
            st.session_state.last_click = now
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# tools
if st.sidebar.button("Clear local CSV"):
    if os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)
        load_data.clear()
        st.sidebar.success("prices.csv removed.")
    else:
        st.sidebar.info("No file to delete.")

st.sidebar.divider()

# Load and display
df = load_data()

# Top layout
col_left, col_right = st.columns([2, 1])

with col_right:
    st.subheader("Latest info")

    if len(df) == 0:
        st.info("No data yet.")
    else:
        last_time = df["time"].iloc[-1]
        last_price = df["price"].iloc[-1]

        st.metric("Last price", f"{last_price:.2f} {cur.upper()}")
        st.write("UTC time:", str(last_time))

        st.write("Rows:", len(df))
        st.write("From:", df["time"].min())
        st.write("To:", df["time"].max())

        st.download_button(
            "Download prices.csv",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="prices.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Strategy")

    strat = st.selectbox("Choose strategy", ["Buy & Hold", "SMA crossover", "Momentum"])

    # parameter
    fast = 5
    slow = 10
    window = 3
    can_run = True

    if strat == "SMA crossover":
        fast = st.slider("SMA fast", 2, 50, 5)
        slow = st.slider("SMA slow", 3, 200, 10)
        if slow <= fast:
            st.warning("Set SMA slow > SMA fast.")
            can_run = False

    if strat == "Momentum":
        window = st.slider("Momentum window", 2, 20, 3)

with col_left:
    st.subheader("Chart and backtest")

    if len(df) < 5:
        st.info("Not enough data yet. Save a few prices.")
    else:
        if not can_run:
            st.info("Strategy not ready (check parameters).")
        else:
            # run strategy
            if strat == "Buy & Hold":
                out = buy_and_hold(df)
            elif strat == "SMA crossover":
                out = sma_crossover(df, fast=fast, slow=slow)
            else:
                out = momentum_strategy(df, window=window)

            # metrics
            tr = total_return(out["strat_value"])
            vol = annual_volatility(out["strat_ret"])
            mdd = max_drawdown(out["strat_value"])
            shrp = sharpe_ratio(out["strat_ret"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total return", f"{tr*100:.4f}%")
            m2.metric("Vol (ann.)", f"{vol*100:.4f}%")
            m3.metric("Max Drawdown", f"{mdd*100:.4f}%")
            m4.metric("Sharpe (rf=0)", f"{shrp:.2f}")

            # plot
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.plot(out["time"], out["price"], label=f"Price ({cur.upper()})")
            if strat == "SMA crossover":
                ax.plot(out["time"], out["sma_fast"], label="SMA fast")
                ax.plot(out["time"], out["sma_slow"], label="SMA slow")

            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel(f"Price ({cur.upper()})")
            ax.grid(True)

            ax2 = ax.twinx()
            ax2.plot(out["time"], out["strat_value"], label="Strategy value")
            ax2.set_ylabel("Cumulative value")

            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")

            st.pyplot(fig)

            st.subheader("Last rows")
            nrows = st.slider("Rows to show", 5, 50, 10)
            st.dataframe(out.tail(nrows), use_container_width=True)
