"""Statistics: Wilson intervals per cell; rliable IQM + stratified bootstrap across seeds x levels."""
import math
import sys

import numpy as np
import pandas as pd


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def _score_matrix(df, method):
    """(n_seeds, n_levels) success rates for one method."""
    sub = df[df.method == method]
    piv = sub.groupby(["seed", "sigma_mm"]).success.mean().unstack("sigma_mm").sort_index(axis=1)
    return piv.to_numpy(dtype=float)


def aggregate(df, reps=5000):
    from rliable import library as rly
    from rliable import metrics
    scores = {m: _score_matrix(df, m) for m in sorted(df.method.unique())}
    point, ci = rly.get_interval_estimates(scores, lambda x: np.array([metrics.aggregate_iqm(x)]), reps=reps)
    return pd.DataFrame([{"method": m, "iqm": point[m][0], "lo": ci[m][0, 0], "hi": ci[m][1, 0]} for m in scores])


def prob_improvement(df, a, b, reps=2000):
    from rliable import library as rly
    from rliable import metrics
    k = f"{a},{b}"
    pair = {k: (_score_matrix(df, a), _score_matrix(df, b))}
    point, ci = rly.get_interval_estimates(pair, metrics.probability_of_improvement, reps=reps)
    return float(np.squeeze(point[k])), float(np.squeeze(ci[k][0])), float(np.squeeze(ci[k][1]))


def per_cell(df):
    g = df.groupby(["method", "sigma_mm"]).success.agg(["sum", "count"]).reset_index()
    g[["lo", "hi"]] = g.apply(lambda r: pd.Series(wilson(r["sum"], r["count"])), axis=1)
    g["rate"] = g["sum"] / g["count"]
    return g


ORDER = ["oracle", "belief", "recurrent", "framestack", "plain"]


def markdown_tables(df, reps=5000, ref="belief"):
    """Report-ready Markdown: per-σ success with Wilson CIs, IQM with bootstrap CIs, P(ref > X)."""
    methods = [m for m in ORDER if m in set(df.method)] + sorted(set(df.method) - set(ORDER))
    sig = sorted(df.sigma_mm.unique())
    cell = per_cell(df).set_index(["method", "sigma_mm"])
    seeds = df.groupby("method").seed.nunique()
    out = ["| method | seeds | " + " | ".join(f"σ={s:g} mm" for s in sig) + " |",
           "|---|---|" + "---|" * len(sig)]
    for m in methods:
        cells = []
        for s in sig:
            r = cell.loc[(m, s)]
            cells.append(f"{r.rate:.2f} [{r.lo:.2f}–{r.hi:.2f}]")
        out.append(f"| {m} | {seeds[m]} | " + " | ".join(cells) + " |")
    agg = aggregate(df, reps=reps).set_index("method")
    out += ["", "| method | IQM success (seeds × σ) | 95% CI |", "|---|---|---|"]
    for m in methods:
        out.append(f"| {m} | {agg.loc[m, 'iqm']:.3f} | {agg.loc[m, 'lo']:.3f}–{agg.loc[m, 'hi']:.3f} |")
    if ref in methods:
        out += ["", f"| comparison | P(improvement) | 95% CI |", "|---|---|---|"]
        for m in methods:
            if m != ref:
                pt, lo, hi = prob_improvement(df, ref, m, reps=reps)
                out.append(f"| P({ref} > {m}) | {pt:.3f} | {lo:.3f}–{hi:.3f} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    df = pd.concat([pd.read_csv(p) for p in sys.argv[1:]])
    print(per_cell(df).to_string(index=False))
    print(aggregate(df).to_string(index=False))
    for m in sorted(df.method.unique()):
        if m != "belief" and "belief" in set(df.method):
            print(f"P(belief > {m}) =", prob_improvement(df, "belief", m))
