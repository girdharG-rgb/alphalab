import numpy as np
import pandas as pd


def sharpe(nav: pd.Series, periods=252) -> float:
    r = nav.pct_change().dropna()
    sd = r.std(ddof=1)
    return float(np.sqrt(periods) * r.mean() / sd) if sd > 0 else 0.0


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    return float((nav / peak - 1.0).min())
