import numpy as np, pandas as pd
from pomdp_pick.eval.stats import wilson, aggregate, prob_improvement

def _fake():
    rows = []
    for method, p in [("a", 0.8), ("b", 0.5)]:
        for seed in range(3):
            for sigma in (2, 10, 30):
                rng = np.random.default_rng(seed * 100 + sigma)
                for ep in range(50):
                    rows.append(dict(method=method, seed=seed, sigma_mm=sigma, episode=ep, success=rng.random() < p))
    return pd.DataFrame(rows)

def test_wilson_bounds():
    lo, hi = wilson(80, 100)
    assert 0.70 < lo < 0.80 < hi < 0.88
    assert wilson(0, 10)[0] == 0.0

def test_aggregate_orders_methods():
    agg = aggregate(_fake())
    a, b = agg.set_index("method").loc["a"], agg.set_index("method").loc["b"]
    assert a["iqm"] > b["iqm"] and a["lo"] <= a["iqm"] <= a["hi"]

def test_probability_of_improvement():
    p, lo, hi = prob_improvement(_fake(), "a", "b")
    assert p > 0.9 and lo <= p <= hi
