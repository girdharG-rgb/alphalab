"""Independent share-based accounting: replays executed dollar trades.

Used as a cross-check on the engine's weight-based bookkeeping.
"""
import numpy as np
import pandas as pd


def replay_ledger(close: pd.DataFrame, trades_usd: pd.DataFrame,
                  cost_bps: float, initial_cash: float) -> pd.Series:
    px = close.to_numpy()
    tr = trades_usd.reindex(index=close.index, columns=close.columns).to_numpy()
    shares = np.zeros(px.shape[1])
    cash = float(initial_cash)
    out = np.empty(len(px))
    for t in range(len(px)):
        shares += tr[t] / px[t]
        cash -= tr[t].sum()
        cash -= cost_bps * 1e-4 * np.abs(tr[t]).sum()
        out[t] = cash + float(shares @ px[t])
    return pd.Series(out, index=close.index)
