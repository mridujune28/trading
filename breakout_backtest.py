#!/usr/bin/env python3
"""
Breakout swing-trade backtest — SIGNAL-EDGE test.

Question this answers: "Does this breakout setup have a positive expectancy
per trade across a large-cap universe, after costs?"  It does NOT yet simulate
a capital-constrained portfolio that can only hold N positions at once and must
rank/pick the top 10 — that is Phase 2, and only worth doing if Phase 1 shows edge.

How it works
------------
- Runs the SAME rule on every stock independently.
- Pools every trade across all stocks.
- Reports per-trade expectancy, win rate, payoff, profit factor, and a
  year-by-year breakdown (so you can see regime dependence, not just one
  flattering aggregate number).

No look-ahead: signals are evaluated on the day's CLOSE; orders fill at the
NEXT bar's OPEN. This matches you reviewing data at night and acting next open.

Data: yfinance (free, adjusted OHLCV). NOTE: this must run somewhere with
internet access to Yahoo (your laptop, Google Colab, or GitHub Actions).
"""

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # only needed when fetching real data

from backtesting import Backtest, Strategy

# ----------------------------- CONFIG --------------------------------------
MARKET = "US"          # "US" or "IN"
START  = "2018-01-01"  # spans 2018 chop, 2020 crash/recovery, 2022 bear, 2023-24 bull
END    = "2025-12-31"

# --- strategy parameters (keep FEW; sweep later, don't curve-fit) ---
BREAKOUT_LOOKBACK = 20     # N-day high that defines the breakout level
VOL_MULT          = 1.5    # breakout-day volume must exceed VOL_MULT * 20d avg
SMA_FAST          = 50
SMA_SLOW          = 200
ATR_LEN           = 14
ATR_INIT_STOP     = 2.0    # initial stop = entry - k*ATR
ATR_TRAIL         = 3.0    # chandelier trail = highest_high_since_entry - k*ATR
MAX_HOLD_DAYS     = 30     # time stop (your 2-30 day horizon)

COMMISSION = 0.001         # 0.1% per side  (slippage + fees proxy; raise for India)
CASH       = 100_000

# Current-ish constituents. NOTE the survivorship caveat in the writeup:
# using today's list historically over-states results because names that
# were dropped from the index (often after underperforming) are missing.
SP100 = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN","AVGO","AXP",
    "BA","BAC","BK","BKNG","BLK","BMY","BRK-B","C","CAT","CHTR","CL","CMCSA","COF",
    "COP","COST","CRM","CSCO","CVS","CVX","DHR","DIS","DOW","DUK","EMR","F","FDX",
    "GD","GE","GILD","GM","GOOG","GOOGL","GS","HD","HON","IBM","INTC","INTU","ISRG",
    "JNJ","JPM","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ","MDT","MET","META",
    "MMM","MO","MRK","MS","MSFT","NEE","NFLX","NKE","NVDA","ORCL","PEP","PFE","PG",
    "PM","PYPL","QCOM","RTX","SBUX","SCHW","SO","SPG","T","TGT","TMO","TMUS","TSLA",
    "TXN","UNH","UNP","UPS","USB","V","VZ","WFC","WMT","XOM",
]

NIFTY50 = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO",
    "BAJFINANCE","BAJAJFINSV","BPCL","BHARTIARTL","BRITANNIA","CIPLA","COALINDIA",
    "DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE",
    "HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC",
    "JSWSTEEL","KOTAKBANK","LT","LTIM","M&M","MARUTI","NESTLEIND","NTPC","ONGC",
    "POWERGRID","RELIANCE","SBILIFE","SBIN","SUNPHARMA","TATACONSUM","TATAMOTORS",
    "TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","WIPRO",
]


# ------------------------- INDICATORS / SIGNALS ----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]

    df["SMA_F"] = c.rolling(SMA_FAST).mean()
    df["SMA_S"] = c.rolling(SMA_SLOW).mean()

    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_LEN).mean()

    # prior N-day high (excludes today -> no look-ahead)
    prior_high = h.rolling(BREAKOUT_LOOKBACK).max().shift(1)
    vol_avg     = v.rolling(BREAKOUT_LOOKBACK).mean().shift(1)

    breakout_day = (c > prior_high) & (v > VOL_MULT * vol_avg)   # day the level breaks
    trend_ok     = (c > df["SMA_S"]) & (df["SMA_F"] > df["SMA_S"])

    # CONFIRMATION: yesterday was a breakout day, today still holds above the
    # level it broke, and trend filter holds today.
    df["EntrySignal"] = (
        breakout_day.shift(1).fillna(False)
        & (c > prior_high.shift(1))
        & trend_ok
    )
    return df


# ----------------------------- STRATEGY ------------------------------------
class BreakoutSwing(Strategy):
    def init(self):
        self.entry_i = None
        self.peak = None
        self.trail = None

    def next(self):
        i = len(self.data) - 1
        price = self.data.Close[-1]
        atr = self.data.ATR[-1]

        if not self.position:
            if bool(self.data.EntrySignal[-1]) and not np.isnan(atr) and atr > 0:
                self.buy()  # fills next open
                self.entry_i = i + 1
                self.peak = price
                self.trail = price - ATR_INIT_STOP * atr
        else:
            self.peak = max(self.peak, self.data.High[-1])
            self.trail = max(self.trail, self.peak - ATR_TRAIL * atr)
            held = i - (self.entry_i if self.entry_i is not None else i)
            if price < self.trail or held >= MAX_HOLD_DAYS:
                self.position.close()  # fills next open


# ------------------------------ RUNNERS ------------------------------------
def fetch(ticker: str) -> pd.DataFrame:
    sym = f"{ticker}.NS" if MARKET == "IN" else ticker
    df = yf.download(sym, start=START, end=END, auto_adjust=True,
                     progress=False, multi_level_index=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def run_one(df: pd.DataFrame):
    df = add_indicators(df).dropna(subset=["SMA_S", "ATR"])
    if len(df) < 250:
        return None
    bt = Backtest(df, BreakoutSwing, cash=CASH, commission=COMMISSION,
                  trade_on_close=False, finalize_trades=True)
    bt.run()
    return bt._results._trades.copy()  # per-trade table


def summarize(trades: pd.DataFrame):
    if trades.empty:
        print("No trades generated."); return
    r = trades["ReturnPct"]  # fraction, net of commission
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    print("="*60)
    print(f"Trades            : {len(r)}")
    print(f"Win rate          : {len(wins)/len(r):.1%}")
    print(f"Avg win           : {wins.mean():.2%}" if len(wins) else "Avg win: n/a")
    print(f"Avg loss          : {losses.mean():.2%}" if len(losses) else "Avg loss: n/a")
    print(f"Payoff (avgW/avgL): {abs(wins.mean()/losses.mean()):.2f}"
          if len(wins) and len(losses) else "Payoff: n/a")
    print(f"EXPECTANCY / trade : {r.mean():.2%}   <-- the number that matters")
    print(f"Profit factor     : {pf:.2f}")
    print(f"Avg hold (bars)    : {trades['Duration'].dt.days.mean():.1f}")
    print("="*60)
    by_year = trades.assign(yr=trades["ExitTime"].dt.year).groupby("yr")["ReturnPct"]
    print("By year  | trades |  expectancy | win rate")
    for yr, g in by_year:
        print(f"  {yr}    |  {len(g):4d}  |   {g.mean():+.2%}  |  {(g>0).mean():.0%}")


def main():
    universe = SP100 if MARKET == "US" else NIFTY50
    all_trades = []
    for k, t in enumerate(universe, 1):
        df = fetch(t)
        if df.empty:
            print(f"  [{k}/{len(universe)}] {t}: no data"); continue
        tr = run_one(df)
        if tr is not None and not tr.empty:
            tr["ticker"] = t
            all_trades.append(tr)
        print(f"  [{k}/{len(universe)}] {t}: {0 if tr is None else len(tr)} trades")
    if all_trades:
        pooled = pd.concat(all_trades, ignore_index=True)
        pooled.to_csv("backtest_trades.csv", index=False)
        summarize(pooled)
    else:
        print("No trades across universe.")


if __name__ == "__main__":
    main()
