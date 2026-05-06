"""Generate publication-ready figures for the updated ASEN 5264 final report.

Uses new training data (3 uncertainties, reduced 13D/16D obs, 5M steps each).
Run from new_paper/ directory:
    python3 gen_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

os.makedirs("figs", exist_ok=True)

SCRIPTS = "../vla_SO-ARM101/src/so_arm101_control/scripts"

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.labelsize":  9,
    "axes.titlesize":  9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi":      200,
    "lines.linewidth": 1.6,
    "axes.grid":       True,
    "grid.alpha":      0.35,
    "grid.linestyle":  "--",
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

PLAIN_COLOR  = "#2166ac"
BELIEF_COLOR = "#d6604d"

def smooth(arr, w=15):
    return np.convolve(arr, np.ones(w) / w, mode="same")

TRIM = 7

# ── Load data ──────────────────────────────────────────────────────────────
sweep       = np.load(f"{SCRIPTS}/results/noise_sweep_reduced_obs.npz")
plain_eval  = np.load(f"{SCRIPTS}/logs/ppo_plain/evaluations.npz")
belief_eval = np.load(f"{SCRIPTS}/logs/ppo_belief/evaluations.npz")
traj        = np.load(f"{SCRIPTS}/logs/ppo_belief/trajectories/trajectory_data_latest.npz",
                      allow_pickle=True)

# ── Fig 1: Noise Sweep ─────────────────────────────────────────────────────
sigmas_mm = sweep["sigmas"] * 1000
plain_s   = sweep["plain_success"] * 100
belief_s  = sweep["belief_success"] * 100

fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.plot(sigmas_mm, plain_s,  "o-",  color=PLAIN_COLOR,  label="Plain PPO",  markersize=4)
ax.plot(sigmas_mm, belief_s, "s--", color=BELIEF_COLOR, label="Belief PPO", markersize=4)
ax.fill_between(sigmas_mm, plain_s, belief_s,
                where=(belief_s >= plain_s), alpha=0.12, color=BELIEF_COLOR,
                label="Belief leads")
ax.fill_between(sigmas_mm, plain_s, belief_s,
                where=(belief_s <  plain_s), alpha=0.12, color=PLAIN_COLOR,
                label="Plain leads")
ax.axvline(15, color="#888", lw=0.8, ls=":", alpha=0.7)
ax.text(15.4, 5, "training\nmax", fontsize=6.5, color="#888", va="bottom")
ax.set_xlabel(r"Observation Noise $\sigma_{\mathrm{ep}}$ (mm)")
ax.set_ylabel("Task Success Rate (%)")
ax.set_xticks(sigmas_mm)
ax.set_xticklabels([str(int(s)) for s in sigmas_mm])
ax.set_ylim(0, 105)
ax.legend(loc="lower left", fontsize=7.5)
fig.tight_layout(pad=0.4)
fig.savefig("figs/noise_sweep.pdf", bbox_inches="tight")
fig.savefig("figs/noise_sweep.png", bbox_inches="tight")
plt.close(fig)
print("Saved figs/noise_sweep.pdf")

# ── Fig 2: Noise sweep — episode length ───────────────────────────────────
plain_len  = np.array([r["mean_ep_len"] for r in []])   # computed below
# Reload with full dict since npz only has summary arrays
# Recompute from sweep npz keys available
# We'll build a secondary axis showing mean ep length from trajectory data
# For now use a bar chart of delta success
delta = belief_s - plain_s
colors = [BELIEF_COLOR if d >= 0 else PLAIN_COLOR for d in delta]
fig, ax = plt.subplots(figsize=(3.4, 2.0))
bars = ax.bar([str(int(s)) for s in sigmas_mm], delta, color=colors, alpha=0.75, edgecolor="k", linewidth=0.5)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel(r"$\sigma_{\mathrm{ep}}$ (mm)")
ax.set_ylabel(r"$\Delta$ Success % (Belief $-$ Plain)")
ax.set_title("Belief PPO advantage by noise level", fontsize=8)
for bar, d in zip(bars, delta):
    ypos = d + 0.5 if d >= 0 else d - 1.5
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{d:+.0f}%",
            ha="center", va="bottom" if d >= 0 else "top", fontsize=6.5)
fig.tight_layout(pad=0.4)
fig.savefig("figs/noise_delta.pdf", bbox_inches="tight")
fig.savefig("figs/noise_delta.png", bbox_inches="tight")
plt.close(fig)
print("Saved figs/noise_delta.pdf")

# ── Fig 3: Learning Curves (reward) ───────────────────────────────────────
plain_steps  = plain_eval["timesteps"]  / 1e6
plain_mean   = plain_eval["results"].mean(axis=1)
belief_steps = belief_eval["timesteps"] / 1e6
belief_mean  = belief_eval["results"].mean(axis=1)

# Find best checkpoints
plain_best_idx  = plain_mean.argmax()
belief_best_idx = belief_mean.argmax()

fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(plain_steps[TRIM:-TRIM],  smooth(plain_mean)[TRIM:-TRIM],
        color=PLAIN_COLOR,  label="Plain PPO")
ax.plot(belief_steps[TRIM:-TRIM], smooth(belief_mean)[TRIM:-TRIM],
        color=BELIEF_COLOR, label="Belief PPO", linestyle="--")
ax.axvline(plain_steps[plain_best_idx],   color=PLAIN_COLOR,  lw=0.9, ls=":",
           alpha=0.8, label=f"Best @ {plain_steps[plain_best_idx]:.2f}M")
ax.axvline(belief_steps[belief_best_idx], color=BELIEF_COLOR, lw=0.9, ls=":",
           alpha=0.8, label=f"Best @ {belief_steps[belief_best_idx]:.2f}M")
ax.set_xlabel("Training Steps (M)")
ax.set_ylabel("Mean Episode Reward")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.legend(loc="lower right", fontsize=7)
fig.tight_layout(pad=0.4)
fig.savefig("figs/learning_curves.pdf", bbox_inches="tight")
fig.savefig("figs/learning_curves.png", bbox_inches="tight")
plt.close(fig)
print("Saved figs/learning_curves.pdf")

# ── Fig 4: Success Rate Curve from trajectory snapshots ───────────────────
import glob

def load_traj_snapshots(base):
    files = glob.glob(f"{base}/trajectory_data_*.npz")
    files = [f for f in files if "latest" not in f]
    files.sort(key=lambda f: int(os.path.basename(f)
                                   .replace("trajectory_data_","")
                                   .replace(".npz","")))
    steps, sr = [], []
    seen = set()
    for f in files:
        d = np.load(f, allow_pickle=True)
        s = int(d["step"])
        if s in seen:
            continue
        seen.add(s)
        steps.append(s / 1e6)
        sr.append(d["successes"].mean() * 100)
    return np.array(steps), np.array(sr)

plain_ts,  plain_tsr  = load_traj_snapshots(f"{SCRIPTS}/logs/ppo_plain/trajectories")
belief_ts, belief_tsr = load_traj_snapshots(f"{SCRIPTS}/logs/ppo_belief/trajectories")

# Only keep up to 5M (the trained range)
mask_p = plain_ts  <= 5.05
mask_b = belief_ts <= 5.05

fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(plain_ts[mask_p],  smooth(plain_tsr[mask_p],  w=5),
        color=PLAIN_COLOR,  label="Plain PPO")
ax.plot(belief_ts[mask_b], smooth(belief_tsr[mask_b], w=5),
        color=BELIEF_COLOR, label="Belief PPO", linestyle="--")
ax.set_xlabel("Training Steps (M)")
ax.set_ylabel("Success Rate (%)")
ax.set_ylim(0, 108)
ax.legend(loc="lower right")
fig.tight_layout(pad=0.4)
fig.savefig("figs/success_curve.pdf", bbox_inches="tight")
fig.savefig("figs/success_curve.png", bbox_inches="tight")
plt.close(fig)
print("Saved figs/success_curve.pdf")

# ── Fig 5: Belief Evolution ────────────────────────────────────────────────
# Pick the episode with longest useful belief evolution (high initial sigma)
ep_lengths = np.asarray(traj["ep_lengths"])
successes  = np.asarray(traj["successes"])
bsig_all   = np.asarray(traj["belief_sigma"])   # (N, 300, 3)
dist_all   = np.asarray(traj["dist_to_block"])  # (N, 300)
wrist_all  = np.asarray(traj["wrist_occluded"]) # (N, 300)
hold_all   = np.asarray(traj["holding"])        # (N, 300)

# Pick a successful episode with meaningful length
cands = [i for i in range(len(successes))
         if successes[i] and ep_lengths[i] > 30]
if not cands:
    cands = list(range(len(successes)))
ep_idx = max(cands, key=lambda i: ep_lengths[i])

T       = int(ep_lengths[ep_idx])
bsig    = bsig_all[ep_idx, :T, :2].mean(axis=-1) * 1000   # XY sigma in mm
dist_mm = dist_all[ep_idx, :T] * 1000
wrist   = wrist_all[ep_idx, :T]
holding = hold_all[ep_idx, :T]
steps   = np.arange(T)

grasp_t = next((t for t in range(1, T) if holding[t] and not holding[t-1]), None)

# Find occlusion spans
occ_starts, occ_ends = [], []
in_occ = False
for t in range(T):
    if wrist[t] and not in_occ:
        occ_starts.append(t); in_occ = True
    elif not wrist[t] and in_occ:
        occ_ends.append(t);   in_occ = False
if in_occ:
    occ_ends.append(T)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.2), sharex=True,
                                gridspec_kw={"hspace": 0.08,
                                             "height_ratios": [1.4, 1]})

for ax in (ax1, ax2):
    for s, e in zip(occ_starts, occ_ends):
        ax.axvspan(s, e, alpha=0.13, color="#888888")

ax1.plot(steps, bsig, color=BELIEF_COLOR, lw=1.5)
ax1.set_ylabel(r"Belief $\sigma_{xy}$ (mm)")
ymax = max(bsig.max() * 1.1, 20)
ax1.set_ylim(0, ymax)
if grasp_t:
    ax1.axvline(grasp_t, color="#2ca02c", lw=1.1, ls=":", label="Grasp")
ax1.legend(loc="upper right", fontsize=7.5,
           handles=[
               mpatches.Patch(color="#888888", alpha=0.35, label="Occluded"),
               plt.Line2D([0],[0], color=BELIEF_COLOR, lw=1.5,
                          label=r"$\sigma_{xy}$"),
               plt.Line2D([0],[0], color="#2ca02c", lw=1.1, ls=":", label="Grasp"),
           ])

ax2.plot(steps, dist_mm, color=PLAIN_COLOR, lw=1.5, label="Dist. to target")
ax2.set_ylabel("EE-to-block (mm)")
ax2.set_xlabel("Timestep")
ax2.set_ylim(0, dist_mm.max() * 1.15)
if grasp_t:
    ax2.axvline(grasp_t, color="#2ca02c", lw=1.1, ls=":")
ax2.legend(loc="upper right", fontsize=7.5)

fig.tight_layout(pad=0.4)
fig.savefig("figs/belief_evolution.pdf", bbox_inches="tight")
fig.savefig("figs/belief_evolution.png", bbox_inches="tight")
plt.close(fig)
print("Saved figs/belief_evolution.pdf")

# ── Fig 6: Episode length vs noise (explore-exploit tension) ─────────────
# Use trajectory snapshot at closest training noise (sigma_ep=0.003..0.015)
# to show episode length relationship - load from sweep if available

# Reconstruct ep_length data from eval files using ep_lengths key
plain_ep_lens  = plain_eval.get("ep_lengths",  None)
belief_ep_lens = belief_eval.get("ep_lengths", None)

if plain_ep_lens is not None and belief_ep_lens is not None:
    # Average ep length in last 3 evals
    p_len_final = plain_ep_lens[-3:].mean()
    b_len_final = belief_ep_lens[-3:].mean()

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    # For noise vs ep_len we only have trajectory snapshots, not sweep
    # Use the sweep data we have - plain vs belief ep_len across noise
    # (not stored in npz — show training curve of ep length instead)
    p_len_curve = plain_eval["ep_lengths"].mean(axis=1)
    b_len_curve = belief_eval["ep_lengths"].mean(axis=1)

    ax.plot(plain_steps[TRIM:-TRIM],  smooth(p_len_curve)[TRIM:-TRIM],
            color=PLAIN_COLOR,  label="Plain PPO")
    ax.plot(belief_steps[TRIM:-TRIM], smooth(b_len_curve)[TRIM:-TRIM],
            color=BELIEF_COLOR, label="Belief PPO", linestyle="--")
    ax.set_xlabel("Training Steps (M)")
    ax.set_ylabel("Mean Episode Length (steps)")
    ax.legend(loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.savefig("figs/ep_length_curve.pdf", bbox_inches="tight")
    fig.savefig("figs/ep_length_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved figs/ep_length_curve.pdf")

print("\nAll figures done.")
