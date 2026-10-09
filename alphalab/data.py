"""Synthetic daily panel used by tests and examples."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Panel:
    close: pd.DataFrame        # dates x tickers, close prices
    adv_usd: pd.DataFrame      # dates x tickers, average daily traded value in USD
    groups: pd.Series          # ticker -> group label (e.g. sector)


def make_panel(n_days=400, n_names=40, n_groups=4, seed=0) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    names = [f"S{i:02d}" for i in range(n_names)]
    mkt = rng.normal(0.0003, 0.008, size=(n_days, 1))
    idio = rng.normal(0.0, 0.015, size=(n_days, n_names))
    drift = rng.normal(0.0002, 0.0003, size=(1, n_names))
    rets = mkt + idio + drift
    p0 = rng.uniform(40, 400, size=(1, n_names))
    close = pd.DataFrame(p0 * np.cumprod(1 + rets, axis=0), index=dates, columns=names)
    base = rng.uniform(8e5, 4e6, size=(1, n_names))
    adv = pd.DataFrame(base * np.exp(rng.normal(0, 0.1, size=(n_days, n_names))),
                       index=dates, columns=names)
    groups = pd.Series([f"G{i % n_groups}" for i in range(n_names)], index=names)
    return Panel(close=close, adv_usd=adv, groups=groups)
