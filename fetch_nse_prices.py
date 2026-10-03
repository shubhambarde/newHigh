"""
Builds the data files for the New Highs & Lows portal: all NSE equities,
last 10 years, one CSV per year in the folder portal_data/.

Each row is one stock on one day where it made at least a 5-day high or low:
date,symbol,close,chg,high_p,low_p
  high_p / low_p = longest period reached (0=5 day, 1=14 day, 2=1 month,
  3=3 month, 4=6 month, 5=1 year), blank if none.

Setup:   pip install yfinance pandas requests
Run:     python fetch_nse_prices.py           full rebuild, 10 years (slow, run once)
         python fetch_nse_prices.py --update  daily refresh, current year only (fast)
Also writes portal_data/years.json, which the website reads.
"""
import io
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}
WINDOWS = [5, 14, 21, 63, 126, 252]      # 5d, 14d, 1m, 3m, 6m, 1y trading days
CHUNK = 100                               # symbols per request batch
OUT_DIR = "portal_data"
UPDATE = "--update" in sys.argv
SHOW_FROM = date(date.today().year, 1, 1) if UPDATE else date.today() - timedelta(days=366 * 10)
FETCH_FROM = SHOW_FROM - timedelta(days=400)            # extra year for the 1y lookback

def get_symbols():
    try:
        r = requests.get(LIST_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        df = df[df["SERIES"].str.strip() == "EQ"]
        syms = sorted(df["SYMBOL"].str.strip().unique())
        open("symbols.txt", "w").write("\n".join(syms))   # cached copy
        return syms
    except Exception as e:
        print("NSE list download failed, using cached symbols.txt:", e)
        return open("symbols.txt").read().split()

def extremes(d):
    """Return high_p / low_p (longest period reached) for each day of one stock."""
    hi = sum((d["high"] >= d["high"].rolling(w, min_periods=w).max()).astype(int) for w in WINDOWS)
    lo = sum((d["low"] <= d["low"].rolling(w, min_periods=w).min()).astype(int) for w in WINDOWS)
    # windows are nested, so the number of windows hit minus one is the longest index
    return hi.where(hi > 0) - 1, lo.where(lo > 0) - 1

def fetch_chunk(symbols):
    tickers = [s + ".NS" for s in symbols]
    data = yf.download(tickers, start=FETCH_FROM.isoformat(), interval="1d",
                       group_by="ticker", auto_adjust=True,
                       threads=True, progress=False)
    out = []
    for s, t in zip(symbols, tickers):
        try:
            d = data[t][["Close", "High", "Low"]].dropna()
        except KeyError:
            continue
        if len(d) < 6:
            continue
        d.columns = ["close", "high", "low"]
        d["chg"] = d["close"].pct_change() * 100
        d["high_p"], d["low_p"] = extremes(d)
        d = d[(d["high_p"].notna() | d["low_p"].notna()) & (d.index.date >= SHOW_FROM)]
        if d.empty:
            continue
        d = d.reset_index().rename(columns={d.index.name or "Date": "date"})
        d["symbol"] = s
        out.append(d[["date", "symbol", "close", "chg", "high_p", "low_p"]])
    return out

def main():
    symbols = get_symbols()
    print(f"{len(symbols)} NSE equity symbols")
    frames = []
    for i in range(0, len(symbols), CHUNK):
        try:
            frames += fetch_chunk(symbols[i:i + CHUNK])
        except Exception as e:
            print("chunk failed, skipping:", e)
        print(f"  {min(i + CHUNK, len(symbols))}/{len(symbols)}")
        time.sleep(1)
    if not frames:
        print("No data fetched")
        return
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["close"] = df["close"].round(2)
    df["chg"] = df["chg"].round(2)
    for c in ("high_p", "low_p"):
        df[c] = df[c].astype("Int64")        # keeps blanks as empty cells
    os.makedirs(OUT_DIR, exist_ok=True)
    for y, g in df.groupby(df["date"].dt.year):
        g = g.sort_values(["date", "symbol"]).copy()
        g["date"] = g["date"].dt.strftime("%Y-%m-%d")
        path = os.path.join(OUT_DIR, f"highs_lows_{y}.csv")
        g.to_csv(path, index=False)
        print(f"Wrote {path}: {len(g)} rows")

    years = sorted(int(f[11:15]) for f in os.listdir(OUT_DIR) if f.startswith("highs_lows_"))
    ist = timezone(timedelta(hours=5, minutes=30))
    json.dump({"years": years, "updated": datetime.now(ist).strftime("%d %b %Y, %H:%M IST")},
              open(os.path.join(OUT_DIR, "years.json"), "w"))

if __name__ == "__main__":
    main()
