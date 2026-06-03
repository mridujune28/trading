#!/usr/bin/env python3
"""
Daily breakout scanner -> static HTML dashboard.

Reuses the SAME signal logic as breakout_backtest.py (imports add_indicators)
so the live scan is identical to what you validated. Run by GitHub Actions
after market close; outputs site/index.html for GitHub Pages.

Two lists:
  - Confirmed entries : breakout held above the level a 2nd day -> buy next open.
  - Pending           : broke out today on volume, not yet confirmed -> watchlist.
Ranked by a transparent composite: relative strength (2x) + momentum + volume.
"""
import os
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

from breakout_backtest import (
    add_indicators, SP100, NIFTY50, MARKET, BREAKOUT_LOOKBACK, VOL_MULT,
    RSI_EXTENDED,
)

INDEX        = "^GSPC" if MARKET == "US" else "^NSEI"
RS_LOOKBACK  = 63    # ~3 months relative strength
MOM_LOOKBACK = 20
TOP_N        = 10
BRAND = dict(purple="#59058F", teal="#00A8A8", blue="#0388BC", navy="#180D5B")


def fetch(ticker: str) -> pd.DataFrame:
    sym = f"{ticker}.NS" if MARKET == "IN" else ticker
    df = yf.download(sym, period="2y", auto_adjust=True,
                     progress=False, multi_level_index=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def index_return(days: int) -> float:
    idx = yf.download(INDEX, period="6mo", auto_adjust=True,
                      progress=False, multi_level_index=False)
    c = idx["Close"].dropna()
    return float(c.iloc[-1] / c.iloc[-days - 1] - 1) if len(c) > days else 0.0


def scan():
    universe = SP100 if MARKET == "US" else NIFTY50
    idx_ret = index_return(RS_LOOKBACK)
    confirmed, pending = [], []
    for t in universe:
        df = fetch(t)
        if len(df) < 220:
            continue
        d = add_indicators(df)
        last = d.iloc[-1]
        c = d["Close"]
        if len(c) <= RS_LOOKBACK:
            continue
        vol20 = d["Volume"].rolling(20).mean().iloc[-1]
        row = dict(
            ticker=t, close=float(last["Close"]),
            rs=float(c.iloc[-1] / c.iloc[-RS_LOOKBACK - 1] - 1) - idx_ret,
            mom=float(c.iloc[-1] / c.iloc[-MOM_LOOKBACK - 1] - 1),
            volsurge=float(last["Volume"] / vol20) if vol20 else 0.0,
            above200=float(last["Close"] / last["SMA_S"] - 1),
            rsi=float(last["RSI"]),
            cmf=float(last["CMF"]),
        )
        if bool(last["EntrySignal"]):
            confirmed.append(row)
        else:
            prior_high = df["High"].rolling(BREAKOUT_LOOKBACK).max().shift(1).iloc[-1]
            broke = last["Close"] > prior_high and last["Volume"] > VOL_MULT * vol20
            trend = last["Close"] > last["SMA_S"] and last["SMA_F"] > last["SMA_S"]
            if broke and trend:
                pending.append(row)
    return rank(confirmed), rank(pending), idx_ret


def rank(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col, w in [("rs", 2.0), ("mom", 1.0), ("volsurge", 1.0), ("cmf", 1.0)]:
        s = df[col]
        sd = s.std() or 1.0
        df[col + "_z"] = (s - s.mean()) / sd * w
    df["score"] = df[[c for c in df.columns if c.endswith("_z")]].sum(axis=1)
    return df.sort_values("score", ascending=False).head(TOP_N).reset_index(drop=True)


def html_table(df: pd.DataFrame, title: str, subtitle: str) -> str:
    if df.empty:
        return f'<h2>{title}</h2><p class="empty">No names today.</p>'
    rows = ""
    for i, r in df.iterrows():
        rsi_cls = "hot" if r.rsi >= RSI_EXTENDED else ("cold" if r.rsi <= 30 else "")
        cmf_cls = "buy" if r.cmf > 0 else "sell"
        rows += (f"<tr><td class='rk'>{i+1}</td><td class='tk'>{r.ticker}</td>"
                 f"<td>{r.close:,.2f}</td><td>{r.rs:+.1%}</td><td>{r.mom:+.1%}</td>"
                 f"<td>{r.volsurge:.1f}x</td>"
                 f"<td class='{rsi_cls}'>{r.rsi:.0f}</td>"
                 f"<td class='{cmf_cls}'>{r.cmf:+.2f}</td>"
                 f"<td>{r.above200:+.1%}</td></tr>")
    return (f'<h2>{title}</h2><p class="sub">{subtitle}</p>'
            '<table><thead><tr><th>#</th><th>Ticker</th><th>Close</th>'
            '<th>RS vs idx (3m)</th><th>Mom (20d)</th><th>Vol surge</th>'
            '<th>RSI</th><th>Money flow</th>'
            f'<th>vs 200DMA</th></tr></thead><tbody>{rows}</tbody></table>')


def build():
    confirmed, pending, idx_ret = scan()
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Breakout Scanner - {MARKET}</title><style>
:root{{--purple:{BRAND['purple']};--teal:{BRAND['teal']};--blue:{BRAND['blue']};--navy:{BRAND['navy']};}}
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f4f5fa;color:var(--navy);margin:0;padding:24px;}}
.head{{background:linear-gradient(120deg,var(--navy),var(--purple));color:#fff;padding:20px 24px;border-radius:14px;}}
.head h1{{margin:0;font-size:20px;}} .head p{{margin:6px 0 0;opacity:.85;font-size:13px;}}
h2{{color:var(--purple);margin:28px 0 2px;font-size:16px;}} .sub{{color:#667;font-size:12px;margin:0 0 8px;}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(24,13,91,.08);}}
th{{background:var(--blue);color:#fff;text-align:right;padding:9px 10px;font-size:12px;}}
th:nth-child(-n+2){{text-align:left;}}
td{{padding:8px 10px;text-align:right;border-top:1px solid #eef;font-size:13px;}}
td.rk,td.tk{{text-align:left;}} td.tk{{font-weight:700;color:var(--navy);}}
.rk{{color:var(--teal);font-weight:700;}}
.hot{{color:#c0392b;font-weight:700;}}    /* RSI extended / overbought - caution */
.cold{{color:var(--blue);font-weight:700;}} /* RSI oversold */
.buy{{color:var(--teal);font-weight:700;}}   /* net buying pressure */
.sell{{color:#c0392b;font-weight:700;}}      /* net selling pressure */
.empty{{color:#889;background:#fff;padding:14px;border-radius:8px;}}
.legend{{font-size:11px;color:#778;margin:6px 0 0;}}
</style></head><body>
<div class="head"><h1>Breakout Scanner &mdash; {MARKET}</h1>
<p>Generated {now} &middot; Index 3m return {idx_ret:+.1%} &middot; ranked by relative strength, momentum, volume &amp; money flow</p></div>
<p class="legend">RSI: <span class="hot">red = extended (&ge;{RSI_EXTENDED}, momentum can still run \u2014 size with care)</span>, <span class="cold">blue = oversold</span>.
Money flow (CMF): <span class="buy">positive = net buying pressure</span>, <span class="sell">negative = net selling</span>. CMF is an OHLCV proxy, not true order-flow.</p>
{html_table(confirmed, "Confirmed entries &mdash; buy candidates next open", "Breakout held above the level a 2nd day. Apply your news/macro overlay before acting.")}
{html_table(pending, "Pending confirmation &mdash; watchlist", "Broke out today on volume but not yet confirmed. Watch for a hold tomorrow.")}
</body></html>"""
    os.makedirs("site", exist_ok=True)
    with open("site/index.html", "w") as f:
        f.write(html)
    print(f"Wrote site/index.html | confirmed={len(confirmed)} pending={len(pending)}")


if __name__ == "__main__":
    build()
