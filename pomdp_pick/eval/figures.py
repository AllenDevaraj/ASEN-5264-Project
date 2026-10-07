"""Figures from evaluation CSVs."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pomdp_pick.eval.stats import per_cell, wilson  # noqa: E402

COLORS = {"oracle": "#2E7D32", "plain": "#8A8F98", "framestack": "#B26B00", "recurrent": "#7B3FA0", "belief": "#0E8FA0"}


def success_vs_sigma(df, path):
    cell = per_cell(df)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for m, g in cell.groupby("method"):
        ax.plot(g.sigma_mm, g.rate * 100, "-o", color=COLORS.get(m), label=m, lw=2, ms=4)
        ax.fill_between(g.sigma_mm, g.lo * 100, g.hi * 100, color=COLORS.get(m), alpha=0.15, lw=0)
    ax.axvline(20, ls=":", color="#999", lw=1)
    ax.text(20.5, 3, "training max", color="#777", fontsize=8)
    ax.set_xlabel("Camera noise σ_ep (mm)")
    ax.set_ylabel("Pick-and-place success (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def learning_curves(runs_root, path, window=1000):
    """Rolling training success vs environment steps; mean across seeds with min-max band."""
    import glob
    from pathlib import Path
    import numpy as np
    runs_root = Path(runs_root)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    finals = []
    methods = sorted({p.parent.parent.parent.name for p in runs_root.glob("*/seed*/monitor/*.csv")})
    for m in methods:
        curves = []
        for sd in sorted((runs_root / m).glob("seed*")):
            fs = sorted(glob.glob(str(sd / "monitor" / "*.csv")))
            if not fs:
                continue
            d = pd.concat([pd.read_csv(f, skiprows=1) for f in fs]).sort_values("t")
            steps = d.l.cumsum().to_numpy()
            succ = d.success.rolling(window, min_periods=max(1, window // 5)).mean().to_numpy()
            curves.append((steps, succ))
        if not curves:
            continue
        grid = np.linspace(0, min(c[0][-1] for c in curves), 200)
        Y = np.array([np.interp(grid, s, np.nan_to_num(y, nan=0.0)) for s, y in curves])
        ax.plot(grid / 1e6, Y.mean(0) * 100, color=COLORS.get(m), lw=2, label=f"{m} (n={len(curves)})")
        ax.fill_between(grid / 1e6, Y.min(0) * 100, Y.max(0) * 100, color=COLORS.get(m), alpha=0.15, lw=0)
        finals.append(dict(method=m, n_seeds=len(curves), final=float(Y[:, -1].mean())))
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel(f"Training success, rolling {window} episodes (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    return pd.DataFrame(finals)


def budget_table(df):
    """Success (Wilson 95% CI) and mean decision latency per planner and simulation budget."""
    rows = []
    # CPU time excludes watchdog SIGSTOP pauses, which wall time would count
    lat = "mean_decision_cpu_s" if "mean_decision_cpu_s" in df else "mean_decision_s"
    for (m, n), g in df.groupby(["method", "n_sims"]):
        k, tot = int(g.success.sum()), len(g)
        lo, hi = wilson(k, tot)
        rows.append(dict(method=m, simulator="learned WM" if m.startswith("wm") else "physics", n_sims=int(n),
                         rate=k / tot, lo=lo, hi=hi, n=tot, decision_s=float(g[lat].mean())))
    return pd.DataFrame(rows).sort_values(["simulator", "n_sims"]).reset_index(drop=True)


def success_vs_budget(df, path, reference=None):
    t = budget_table(df)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=150)
    for (sim, g), c in zip(t.groupby("simulator"), ("#C2185B", "#3949AB")):
        a1.errorbar(g.n_sims, g.rate * 100, yerr=[(g.rate - g.lo) * 100, (g.hi - g.rate) * 100],
                    fmt="-o", color=c, label=sim, lw=2, ms=4, capsize=3)
        a2.plot(g.n_sims, g.decision_s, "-o", color=c, label=sim, lw=2, ms=4)
    if reference is not None:
        a1.axhline(reference[1] * 100, ls="--", color=COLORS["belief"], lw=1.5, label=reference[0])
    for a in (a1, a2):
        a.set_xscale("log", base=2)
        a.set_xlabel("Simulations per decision")
        a.grid(alpha=0.3)
        a.legend(frameon=False, fontsize=8)
    a1.set_ylabel("Pick-and-place success (%)")
    a1.set_ylim(0, 100)
    a2.set_ylabel("CPU time per decision (s)")
    a2.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    df = pd.concat([pd.read_csv(p) for p in sys.argv[2:]])
    success_vs_sigma(df, sys.argv[1])
