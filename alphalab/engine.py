"""Daily close-to-close backtest engine with participation-capped execution.

Timeline for day t (all trading happens at the close):
  1. Positions held from the close of t-1 earn the close-to-close return r_t.
  2. NAV is marked and weights drift with relative performance.
  3. We trade toward target[t]; each name's trade is limited to
     `participation * adv_usd[t]` dollars. Unfilled quantity is NOT carried
     forward explicitly: tomorrow's target is simply traded against whatever
     is held.
  4. Trading costs (bps of dollars traded) are deducted from NAV.
Weights are always expressed as fractions of current NAV.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    nav: pd.Series
    weights: pd.DataFrame      # post-trade weights, fraction of post-cost NAV
    trades_usd: pd.DataFrame   # signed dollars traded at each close
    costs_usd: pd.Series
    gross_pnl: pd.Series       # fraction of previous NAV


def run_backtest(close, targets, adv_usd, participation=0.1, cost_bps=5.0,
                 initial_nav=1e7) -> BacktestResult:
    dates, names = close.index, close.columns
    rets = close.pct_change().fillna(0.0).to_numpy()
    tgt = targets.reindex(index=dates, columns=names).fillna(0.0).to_numpy()
    adv = adv_usd.reindex(index=dates, columns=names).to_numpy()
    bps = cost_bps * 1e-4

    T, N = rets.shape
    nav = initial_nav
    w = np.zeros(N)
    nav_path = np.empty(T)
    w_path = np.zeros((T, N))
    trades = np.zeros((T, N))
    costs = np.zeros(T)
    gpnl = np.zeros(T)

    for t in range(T):
        r = rets[t]
        gross = float(w @ r)
        nav = nav * (1.0 + gross)
        w = w * (1.0 + r)

        cap_w = participation * adv[t] / initial_nav
        delta = np.clip(tgt[t] - w, -cap_w, cap_w)

        cost = bps * np.abs(delta).sum()
        trades[t] = delta * nav
        costs[t] = cost * nav
        nav = nav * (1.0 - cost)
        w = w + delta

        nav_path[t] = nav
        w_path[t] = w
        gpnl[t] = gross

    return BacktestResult(
        nav=pd.Series(nav_path, index=dates),
        weights=pd.DataFrame(w_path, index=dates, columns=names),
        trades_usd=pd.DataFrame(trades, index=dates, columns=names),
        costs_usd=pd.Series(costs, index=dates),
        gross_pnl=pd.Series(gpnl, index=dates),
    )
