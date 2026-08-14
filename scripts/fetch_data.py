#!/usr/bin/env python3
"""Download a daily close series for the bootstrap world.

    python scripts/fetch_data.py SPY          -> data/SPY.csv (date, close)

Run on a machine with internet access; the bench itself never fetches.
"""

import os
import sys


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    import yfinance as yf
    df = yf.download(ticker, start="2000-01-01", auto_adjust=True,
                     progress=False)
    if df.empty:
        sys.exit(f"no data returned for {ticker}")
    out = df["Close"].reset_index()
    out.columns = ["date", "close"]
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", f"{ticker}.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} rows, {out['date'].min().date()} "
          f"to {out['date'].max().date()})")


if __name__ == "__main__":
    main()
