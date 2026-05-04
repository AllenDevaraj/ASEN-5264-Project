"""Regenerate system_pipeline figure."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

os.makedirs("figs", exist_ok=True)

plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    9,
    "axes.grid":    False,
    "pdf.fonttype": 42,
})

PLAIN_COL  = "#2166ac"
BELIEF_COL = "#d6604d"

C_ENV = dict(facecolor="#ddeeff", edgecolor="#1a5a9a", linewidth=1.2)
C_OBS = dict(facecolor="#fffadd", edgecolor="#a07800", linewidth=1.2)
C_PF  = dict(facecolor="#f0e0ff", edgecolor="#6a1a8a", linewidth=1.2)
C_POL = dict(facecolor="#dff5e3", edgecolor="#1e6b2e", linewidth=1.2)

# ── figure with two axes ────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 4.8),
                                  gridspec_kw={"wspace": 0.10})
for ax in (ax_a, ax_b):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

ax_a.set_title("(a)  Plain PPO — Baseline",
               fontsize=9, fontweight="bold", pad=8)
ax_b.set_title("(b)  Belief-Augmented PPO",
               fontsize=9, fontweight="bold", pad=8)


def place_block(ax, y, line1, line2, style):
    """
    Drop a text box.  Returns the Text artist so callers can
    measure its bounding box after fig.draw() if needed.
    """
    return ax.text(
        0.50, y,
        line1 + "\n" + line2,
        ha="center", va="center",
        fontsize=8.5, linespacing=1.55,
        transform=ax.transAxes, zorder=3,
        bbox=dict(boxstyle="round,pad=0.45", **style),
    )


def arrow_between(ax, y_from, y_to, lbl=""):
    """Straight down arrow with an optional right-side label."""
    ax.annotate(
        "", xy=(0.50, y_to), xytext=(0.50, y_from),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=0.95,
                        color="#333333", mutation_scale=11),
        zorder=4,
    )
    if lbl:
        mid = (y_from + y_to) / 2
        ax.text(0.58, mid, lbl,
                ha="left", va="center",
                fontsize=7.5, color="#444",
                transform=ax.transAxes, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def feedback_loop(ax, y_env, y_pol, col):
    """
    L-shaped feedback: left edge of Policy → gutter → left edge of Env.
    The horizontal segments meet the left edges of the boxes;
    the arrowhead points right into the Env box.
    xl  = x of the vertical gutter line
    xe  = x of the box left edges  (set conservatively far left)
    """
    xl = 0.05   # gutter (always clear of boxes)
    xe = 0.30   # approximate left edge of the centred boxes

    # Draw the three-segment path
    segs_x = [xe,   xl,   xl,   xe  ]
    segs_y = [y_pol, y_pol, y_env, y_env]
    ax.plot(segs_x, segs_y,
            color=col, lw=1.0,
            transform=ax.transAxes,
            solid_capstyle="round", solid_joinstyle="round",
            zorder=2, clip_on=False)

    # Arrowhead entering Env box from the left
    ax.annotate(
        "", xy=(xe, y_env), xytext=(xe - 0.025, y_env),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.0,
                        color=col, mutation_scale=10),
        zorder=5,
    )

    # Label sits on the LEFT of the vertical gutter  ← no box alignment clash
    ax.text(xl - 0.015, (y_pol + y_env) / 2,
            r"$a_t \in \mathrm{R}^{6}$",
            ha="right", va="center",
            fontsize=8.5, color=col,
            transform=ax.transAxes, zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


# ── spacing constants ───────────────────────────────────────────────────────
# Boxes auto-size to their text.  Estimated half-height with pad ≈ 0.063.
# Arrows start/end 0.07 away from box centre so they clear the box visually.
GAP = 0.07

# ── (a) Plain PPO ──────────────────────────────────────────────────────────
# Box centres:  Env=0.82   Obs=0.53   Policy=0.24
# Spacings:     0.29       0.29

place_block(ax_a, 0.82, "MuJoCo Env",        "SO-ARM101",            C_ENV)
arrow_between(ax_a, 0.82 - GAP, 0.53 + GAP, r"$o_t$  (noisy)")
place_block(ax_a, 0.53, "15-D Observation",  "noisy pose, joints",   C_OBS)
arrow_between(ax_a, 0.53 - GAP, 0.24 + GAP)
place_block(ax_a, 0.24, "MLP Policy (PPO)",  "[256, 256] tanh",      C_POL)

feedback_loop(ax_a, y_env=0.82, y_pol=0.24, col=PLAIN_COL)

# ── (b) Belief PPO ─────────────────────────────────────────────────────────
# Box centres:  Env=0.87   PF=0.67   Obs=0.47   Policy=0.27
# Spacings:     0.20       0.20       0.20

place_block(ax_b, 0.87, "MuJoCo Env",        "SO-ARM101",            C_ENV)
arrow_between(ax_b, 0.87 - GAP, 0.67 + GAP, r"$o_t$  (noisy / null)")
place_block(ax_b, 0.67, "Particle Filter",   "N = 300 particles",    C_PF)
arrow_between(ax_b, 0.67 - GAP, 0.47 + GAP, r"$\mu_b,\;\sigma_b$")
place_block(ax_b, 0.47, "18-D Observation",  "joints, belief mean/std", C_OBS)
arrow_between(ax_b, 0.47 - GAP, 0.27 + GAP)
place_block(ax_b, 0.27, "MLP Policy (PPO)",  "[256, 256] tanh",      C_POL)

feedback_loop(ax_b, y_env=0.87, y_pol=0.27, col=BELIEF_COL)

# centre divider
fig.add_artist(plt.Line2D(
    [0.50, 0.50], [0.04, 0.96],
    transform=fig.transFigure,
    color="#cccccc", lw=0.8, linestyle="--"))

fig.savefig("figs/system_pipeline.pdf", bbox_inches="tight", pad_inches=0.06)
fig.savefig("figs/system_pipeline.png", bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
print("Saved figs/system_pipeline.pdf / .png")
