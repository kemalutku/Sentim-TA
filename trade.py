# trade.py ── safe & numerically-stable back-tester ──────────────────
from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


# ─────────────────────  trading simulator  ────────────────────── #
def trading(
    timestamps,
    closes,
    predictions,
    initial_fund: float = 100.0,
    commission: float = 0.01,
):
    from datetime import datetime
    import numpy as np
    import pandas as pd

    # 1) prep timestamps → datetime
    if isinstance(timestamps[0], (int, float, np.integer, np.floating)):
        timestamps = [datetime.utcfromtimestamp(int(ts) / 1000) for ts in timestamps]

    capital, shares = initial_fund, 0.0
    last_buy_price = last_buy_date = prev_label = None
    daily_equity, trades = [], []

    for ts, close, pred in zip(timestamps, closes, predictions):
        daily_equity.append(capital if shares == 0 else capital + shares * close)

        if pred == prev_label:
            continue

        if pred == 1 and shares == 0:        # BUY
            shares = (capital - commission) / close
            capital, last_buy_price, last_buy_date = 0, close, ts

        elif pred == 2 and shares > 0:       # SELL
            gross = shares * close
            net   = gross - commission
            trades.append(
                dict(
                    buy_date=last_buy_date,
                    sell_date=ts,
                    buy_price=last_buy_price,
                    sell_price=close,
                    profit_pct=(close - last_buy_price) / last_buy_price * 100,
                    length=(ts - last_buy_date).days,
                )
            )
            capital, shares = net, 0
            last_buy_price = last_buy_date = None

        prev_label = pred

    # final portfolio value
    last_close  = closes[-1]
    final_value = capital if shares == 0 else capital + shares * last_close
    buy_hold    = (initial_fund / closes[0]) * last_close

    # equity curve metrics
    equity = pd.Series(daily_equity)
    daily_ret = equity.pct_change().fillna(0)
    sharpe = daily_ret.mean() / daily_ret.std() if daily_ret.std() else 0

    total_days = max((timestamps[-1] - timestamps[0]).days, 1)
    ann_return = safe_cagr(final_value, initial_fund, total_days)  # ✔ no overflow

    trade_pcts = [t["profit_pct"] for t in trades]
    idle_ratio = predictions.count(0) / len(predictions) * 100

    result = {
        "Final Value": final_value,
        "Buy & Hold": buy_hold,
        "CAGR (%)": ann_return,
        "Sharpe Ratio (daily)": sharpe,
        "Trades Performed": len(trades),
        "Trades Won": sum(p > 0 for p in trade_pcts),
        "Win Rate (%)": (sum(p > 0 for p in trade_pcts) / len(trades) * 100) if trades else 0,
        "Avg Profit/Trade (%)": np.mean(trade_pcts) if trades else 0,
        "Avg Trade Length (days)": np.mean([t["length"] for t in trades]) if trades else 0,
        "Max Profit/Trade (%)": max(trade_pcts, default=0),
        "Max Loss/Trade (%)": min(trade_pcts, default=0),
        "Idle Ratio (%)": idle_ratio,
        "Period (days)": total_days,
    }
    return result, pd.DataFrame(trades)

def safe_cagr(final_val: float, start_val: float, days: int) -> float:
    """
    Log-based CAGR, avoids overflow and invalid values.
    Returns percentage (e.g. 12.3 for 12.3 %).
    """
    if days <= 0 or start_val <= 0 or final_val <= 0:
        return 0.0
    years = days / 365.0
    if final_val / start_val > 1e12:
        a = 4
    return (np.exp(np.log(final_val / start_val) / years) - 1.0) * 100
