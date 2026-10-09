import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from alphalab import make_panel, momentum_signal, build_targets, run_backtest


@pytest.fixture(scope="module")
def setup():
    p = make_panel()
    tg = build_targets(momentum_signal(p.close), p.groups)
    return p, tg
