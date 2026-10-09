import numpy as np
from alphalab import run_backtest, momentum_signal, build_targets, sharpe
from alphalab.signals import cs_rank, cs_zscore


def test_zscore_rows():
    import pandas as pd
    x = pd.DataFrame(np.random.default_rng(1).normal(size=(5, 10)))
    z = cs_zscore(x)
    assert np.allclose(z.mean(axis=1), 0) and np.allclose(z.std(axis=1, ddof=0), 1)


def test_rank_range():
    import pandas as pd
    x = pd.DataFrame(np.random.default_rng(2).normal(size=(3, 8)))
    r = cs_rank(x)
    assert r.max().max() == 1.0 and r.min().min() > 0


def test_signal_no_lookahead(setup):
    p, _ = setup
    s_full = momentum_signal(p.close)
    s_trunc = momentum_signal(p.close.iloc[:200])
    assert np.allclose(s_full.iloc[:200].fillna(0), s_trunc.fillna(0))


def test_targets_neutral_and_gross(setup):
    p, tg = setup
    live = tg[tg.abs().sum(axis=1) > 0]
    assert np.allclose(live.sum(axis=1), 0, atol=1e-12)
    assert np.allclose(live.abs().sum(axis=1), 1.0)
    for g in p.groups.unique():
        cols = p.groups.index[p.groups == g]
        assert np.allclose(live[cols].sum(axis=1), 0, atol=1e-12)


def test_engine_runs_and_nav_positive(setup):
    p, tg = setup
    res = run_backtest(p.close, tg, p.adv_usd)
    assert (res.nav > 0).all()
    assert np.isfinite(sharpe(res.nav))


def test_no_trade_when_flat_targets(setup):
    p, tg = setup
    res = run_backtest(p.close, tg * 0.0, p.adv_usd)
    assert np.allclose(res.nav, 1e7) and res.trades_usd.abs().sum().sum() == 0
