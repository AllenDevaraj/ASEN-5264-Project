"""Figures from evaluation CSVs."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pomdp_pick.eval.stats import per_cell  # noqa: E402

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


if __name__ == "__main__":
    df = pd.concat([pd.read_csv(p) for p in sys.argv[2:]])
    success_vs_sigma(df, sys.argv[1])
