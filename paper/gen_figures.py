"""Generate publication-ready figures for the ASEN 5264 final report."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

os.makedirs("figs", exist_ok=True)

# ── Global style (IEEE serif, 9 pt) ────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "legend.fontsize":    8,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "figure.dpi":         200,
    "lines.linewidth":    1.6,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "pdf.fonttype":       42,   # embed fonts as Type 42 (TrueType)
    "ps.fonttype":        42,
})

PLAIN_COLOR  = "#2166ac"
BELIEF_COLOR = "#d6604d"

# ───────────────────────────────────────────────────────────────────────────
# Load shared data
# ───────────────────────────────────────────────────────────────────────────
sweep       = np.load("../vla_SO-ARM101/src/so_arm101_control/scripts/"
                      "results/noise_sweep_latest.npz")
plain_eval  = np.load("../vla_SO-ARM101/src/so_arm101_control/scripts/"
                      "logs/ppo_plain/evaluations.npz")
belief_eval = np.load("../vla_SO-ARM101/src/so_arm101_control/scripts/"
                      "logs/ppo_belief/evaluations.npz")

def smooth(arr, w=15):
    return np.convolve(arr, np.ones(w) / w, mode="same")

TRIM = 7

# ── Fig 1: Noise Sweep ─────────────────────────────────────────────────────
sigmas_mm = sweep["sigmas"] * 1000
plain_s   = sweep["plain_success"] * 100
belief_s  = sweep["belief_success"] * 100

fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(sigmas_mm, plain_s,  "o-",  color=PLAIN_COLOR,  label="Plain PPO",  markersize=4)
ax.plot(sigmas_mm, belief_s, "s--", color=BELIEF_COLOR, label="Belief PPO", markersize=4)
ax.fill_between(sigmas_mm, plain_s, belief_s,
                where=(belief_s >= plain_s), alpha=0.12, color=BELIEF_COLOR)
ax.fill_between(sigmas_mm, plain_s, belief_s,
                where=(belief_s <  plain_s), alpha=0.12, color=PLAIN_COLOR)
ax.set_xlabel(r"Observation Noise $\sigma$ (mm)")
ax.set_ylabel("Task Success Rate (%)")
ax.set_xticks(sigmas_mm)
ax.set_xticklabels([str(int(s)) for s in sigmas_mm])
ax.set_ylim(0, 85)
ax.legend(loc="upper right")
fig.tight_layout(pad=0.4)
fig.savefig("figs/noise_sweep.pdf", bbox_inches="tight")
fig.savefig("figs/noise_sweep.png", bbox_inches="tight")
plt.close(fig)
print("Saved figs/noise_sweep.pdf")

# ── Fig 2: Learning Curves ─────────────────────────────────────────────────
plain_steps  = plain_eval["timesteps"]  / 1e6
plain_mean   = plain_eval["results"].mean(axis=1)
belief_steps = belief_eval["timesteps"] / 1e6
belief_mean  = belief_eval["results"].mean(axis=1)

fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(plain_steps[TRIM:-TRIM],  smooth(plain_mean)[TRIM:-TRIM],
        color=PLAIN_COLOR,  label="Plain PPO")
ax.plot(belief_steps[TRIM:-TRIM], smooth(belief_mean)[TRIM:-TRIM],
        color=BELIEF_COLOR, label="Belief PPO", linestyle="--")
ax.set_xlabel("Training Steps (M)")
ax.set_ylabel("Mean Episode Reward")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.legend(loc="lower right")
fig.tight_layout(pad=0.4)
fig.savefig("figs/learning_curves.pdf", bbox_inches="tight")
fig.savefig("figs/learning_curves.png", bbox_inches="tight")
plt.close(fig)
print("Saved figs/learning_curves.pdf")

# ── Fig 3: Success Rate Curve ──────────────────────────────────────────────
TIMEOUT  = 195
plain_sr  = (plain_eval["ep_lengths"]  < TIMEOUT).mean(axis=1) * 100
belief_sr = (belief_eval["ep_lengths"] < TIMEOUT).mean(axis=1) * 100

fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(plain_steps[TRIM:-TRIM],  smooth(plain_sr)[TRIM:-TRIM],
        color=PLAIN_COLOR,  label="Plain PPO")
ax.plot(belief_steps[TRIM:-TRIM], smooth(belief_sr)[TRIM:-TRIM],
        color=BELIEF_COLOR, label="Belief PPO", linestyle="--")
ax.set_xlabel("Training Steps (M)")
ax.set_ylabel("Success Rate (%)")
ax.set_ylim(0, 105)
ax.legend(loc="lower right")
fig.tight_layout(pad=0.4)
fig.savefig("figs/success_curve.pdf", bbox_inches="tight")
fig.savefig("figs/success_curve.png", bbox_inches="tight")
plt.close(fig)
print("Saved figs/success_curve.pdf")
plt.rcParams["axes.grid"] = True
# system_pipeline is generated separately by gen_pipeline.py

# ── Fig 5: Belief Evolution (two-panel, ep19 of latest checkpoint) ─────────
traj     = np.load("../vla_SO-ARM101/src/so_arm101_control/scripts/"
                   "logs/ppo_belief/trajectories/trajectory_data_latest.npz",
                   allow_pickle=True)
ep_idx   = 19
T        = int(np.asarray(traj["ep_lengths"])[ep_idx])
bsig     = np.asarray(traj["belief_sigma"])[ep_idx, :T].mean(axis=-1) * 1000
dist_mm  = np.asarray(traj["dist_to_block"])[ep_idx, :T] * 1000
wrist_oc = np.asarray(traj["wrist_occluded"])[ep_idx, :T]
holding  = np.asarray(traj["holding"])[ep_idx, :T]
steps    = np.arange(T)

grasp_t  = next((t for t in range(1, T) if holding[t] and not holding[t-1]), None)
occ_end  = int(np.where(~wrist_oc)[0][0]) if (~wrist_oc).any() else T

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.0), sharex=True,
                                gridspec_kw={"hspace": 0.08,
                                             "height_ratios": [1.5, 1]})

for ax in (ax1, ax2):
    ax.axvspan(0, occ_end, alpha=0.13, color="#888888", label="Occlusion")

ax1.plot(steps, bsig, color=BELIEF_COLOR, lw=1.5, label=r"Belief $\sigma_b$")
ax1.set_ylabel(r"Belief $\sigma_b$ (mm)")
ax1.set_ylim(0, 82)
if grasp_t:
    ax1.axvline(grasp_t, color="#2ca02c", lw=1.1, ls=":", label="Grasp")
ax1.legend(loc="upper left", fontsize=7.5,
           handles=[
               mpatches.Patch(color="#888888", alpha=0.35, label="Occlusion"),
               plt.Line2D([0],[0], color=BELIEF_COLOR, lw=1.5,
                          label=r"Belief $\sigma_b$"),
               plt.Line2D([0],[0], color="#2ca02c", lw=1.1, ls=":",
                          label="Grasp"),
           ])
ax1.text(2, 76, "← occlusion (wrist camera)", fontsize=7, color="#666")
ax1.text(occ_end + 2, 58, "↓ approach\n  reduces $\\sigma$", fontsize=7,
         color="#444", va="top")

ax2.plot(steps, dist_mm, color=PLAIN_COLOR, lw=1.5,
         label="Dist.\ to target")
ax2.set_ylabel("Dist. to target (mm)")
ax2.set_xlabel("Timestep")
ax2.set_ylim(0, 165)
if grasp_t:
    ax2.axvline(grasp_t, color="#2ca02c", lw=1.1, ls=":")
ax2.legend(loc="upper right", fontsize=7.5)

fig.tight_layout(pad=0.4)
fig.savefig("figs/belief_evolution.pdf", bbox_inches="tight")
fig.savefig("figs/belief_evolution.png", bbox_inches="tight")
plt.close(fig)
print("Saved figs/belief_evolution.pdf")

print("\nAll figures done.")
