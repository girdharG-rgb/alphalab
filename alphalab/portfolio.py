"""Turn raw signals into target portfolio weights."""
import numpy as np
import pandas as pd


def group_neutralize(sig: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    out = sig.copy()
    for g, members in groups.groupby(groups).groups.items():
        cols = list(members)
        out[cols] = sig[cols].sub(sig[cols].mean(axis=1), axis=0)
    return out


def build_targets(sig: pd.DataFrame, groups: pd.Series, gross: float = 1.0) -> pd.DataFrame:
    """Group-neutral, dollar-neutral target weights scaled to given gross exposure.

    Weights are fractions of NAV. Rows with no valid signal become all-zero.
    """
    w = group_neutralize(sig.fillna(0.0), groups)
    denom = w.abs().sum(axis=1).replace(0.0, np.nan)
    w = w.div(denom, axis=0).mul(gross).fillna(0.0)
    return w
