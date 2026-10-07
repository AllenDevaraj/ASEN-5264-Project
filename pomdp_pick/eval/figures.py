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
