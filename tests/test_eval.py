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


def _fake_eval(rng):
    import pandas as pd
    rows = []
    for m, p in (("belief", 0.6), ("plain", 0.2)):
        for seed in (0, 1, 2):
            for s in (2, 10, 30):
                for ep in range(20):
                    rows.append(dict(method=m, seed=seed, sigma_mm=s, episode=ep, success=rng.random() < p))
    return pd.DataFrame(rows)


def test_markdown_tables_report_per_sigma_rates_and_iqm():
    import numpy as np
    from pomdp_pick.eval.stats import markdown_tables
    md = markdown_tables(_fake_eval(np.random.default_rng(0)), reps=200)
    assert "| belief |" in md and "| plain |" in md and "IQM" in md and "P(belief > plain)" in md
    assert md.count("\n|") >= 6


def test_learning_curves_figure_from_monitor_csvs(tmp_path):
    import numpy as np
    from pomdp_pick.eval.figures import learning_curves
    rng = np.random.default_rng(0)
    for m in ("belief", "plain"):
        for s in (0, 1):
            d = tmp_path / m / f"seed{s}" / "monitor"
            d.mkdir(parents=True)
            lines = ['#{"t_start": 0}', "r,l,t,success,attempts"]
            lines += [f"0,{rng.integers(10, 100)},{i},{float(rng.random() < 0.5)},1" for i in range(300)]
            (d / "w0.monitor.csv").write_text("\n".join(lines) + "\n")
    final = learning_curves(tmp_path, tmp_path / "lc.png", window=50)
    assert set(final.method) == {"belief", "plain"} and (tmp_path / "lc.png").stat().st_size > 1000
    assert final.n_seeds.tolist() == [2, 2]
