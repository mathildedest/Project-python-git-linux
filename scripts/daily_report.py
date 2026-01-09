import os
import pandas as pd
import numpy as np

DATA_PATH = "data/prices.csv"

def max_drawdown(values):
    roll_max = values.cummax()
    dd = values / roll_max - 1.0
    return float(dd.min())

def main():
    # create folder
    os.makedirs("reports", exist_ok=True)

    # if no data, stop
    if not os.path.exists(DATA_PATH):
        print("No data/prices.csv found. No report generated.")
        return

    #load data
    df = pd.read_csv(DATA_PATH)

    #clean colums
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["time", "price"]).sort_values("time").reset_index(drop=True)

    if len(df) < 2:
        print("Not enough data to compute stats.")
        return

    #today
    today_date = pd.Timestamp.utcnow().date()
    today_str = today_date.strftime("%Y-%m-%d")
    #data
    df_today = df[df["time"].dt.date == today_date].copy()

     # open/close for today
    if len(df_today) >= 1:
        open_today = float(df_today["price"].iloc[0])
        close_today = float(df_today["price"].iloc[-1])
        today_obs = int(len(df_today))
        daily_return = (close_today / open_today - 1.0) if open_today != 0 else 0.0
    else:
        open_today = None
        close_today = None
        today_obs = 0
        daily_return = None

    # full series stats (simple)
    df["ret"] = df["price"].pct_change()
    ret = df["ret"].dropna()

    if len(ret) >= 2:
        vol_ann = float(ret.std() * np.sqrt(365))
    else:
        vol_ann = 0.0

    df["ret"] = df["ret"].fillna(0.0)
    df["value"] = (1.0 + df["ret"]).cumprod()
    mdd = max_drawdown(df["value"])

    # last price info
    last_time = df["time"].iloc[-1]
    last_price = float(df["price"].iloc[-1])

    out_path = f"reports/report_{today_str}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Daily Report\n")
        f.write("\n")
        f.write(f"Date: {today_str}\n")

        f.write("DATA\n")
        f.write(f"Observations: {len(df)}\n")
        f.write(f"Open: {last_time}\n")
        f.write(f"Close: {last_price}\n")

        f.write("Today\n")
        f.write(f"- Observations today: {today_obs}\n")
        if open_today is None:
            f.write("- Open today: no data today\n")
            f.write("- Close today: no data today\n")
            f.write("- Daily return: NA\n\n")
        else:
            f.write(f"- Open today: {open_today}\n")
            f.write(f"- Close today: {close_today}\n")
            f.write(f"- Daily return: {daily_return}\n\n")

        f.write("METRICS\n")
        f.write(f"- Volatility (annualized): {vol_ann}\n")
        f.write(f"- Max Drawdown (buy&hold): {mdd}\n")

    print(f"Report saved: {out_path}")

if __name__ == "__main__":
    main()
