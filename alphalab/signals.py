"""Cross-sectional signal operators."""
import numpy as np
import pandas as pd


def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank in (0, 1], NaN-preserving."""
    return x.rank(axis=1, pct=True)


def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    mu = x.mean(axis=1)
    sd = x.std(axis=1, ddof=0).replace(0.0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0)


def ts_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).mean()


def momentum_signal(close: pd.DataFrame, lookback=60, skip=5, smooth=3) -> pd.DataFrame:
    """Skip-month style momentum: return over [t-lookback, t-skip], smoothed.

    The signal at row t only uses closes up to and including day t.
    """
    mom = close.shift(skip) / close.shift(lookback) - 1.0
    sig = cs_zscore(ts_mean(mom, smooth))
    return sig
