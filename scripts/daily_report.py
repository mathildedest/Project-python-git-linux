import os
import pandas as pd
import numpy as np

DATA_PATH = "data/prices.csv"

def max_drawdown(values):
    roll_max = values.cummax()
    dd = values / roll_max - 1.0
    return float(dd.min())

def main():
    # Create reports folder if needed
    os.makedirs("reports", exist_ok=True)

    # If no data, stop
    if not os.path.exists(DATA_PATH):
        print("No data/prices.csv found. No report generated.")
        return

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).sort_values("time")

    if len(df) < 2:
        print("Not enough data to compute stats.")
        return

    # Simple daily stats on full file (student version)
    df["ret"] = df["price"].pct_change()
    vol = float(df["ret"].std() * np.sqrt(365)) if df["ret"].std() == df["ret"].std() else 0.0

    open_p = float(df["price"].iloc[0])
    close_p = float(df["price"].iloc[-1])

    df["ret"] = df["ret"].fillna(0.0)
    df["value"] = (1 + df["ret"]).cumprod()
    mdd = max_drawdown(df["value"])

    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    out_path = f"reports/report_{today}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Daily Report (UTC)\n")
        f.write("-----------------\n")
        f.write(f"Date: {today}\n")
        f.write(f"Observations: {len(df)}\n")
        f.write(f"Open: {open_p}\n")
        f.write(f"Close: {close_p}\n")
        f.write(f"Volatility (annualized, simple): {vol}\n")
        f.write(f"Max Drawdown (buy&hold, simple): {mdd}\n")

    print(f"Report saved: {out_path}")

if __name__ == "__main__":
    main()
