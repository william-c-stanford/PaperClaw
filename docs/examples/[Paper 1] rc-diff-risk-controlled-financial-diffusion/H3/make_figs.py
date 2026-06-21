"""Generate figures summarising H3 results."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("h3_results_v3.json") as f:
    R = json.load(f)

# fig1: AUC per metric x corruption (paired)
metrics_order = [
    "KS","Wasserstein1","TailHillErr","VaR5Err","ES5Err","MMD",
    "AbsACFMaha","SignACFMaha",
    "SigW_L2_d2","SigW_L2_d3","SigW_L2_d4",
    "SigW_Maha_d2","SigW_Maha_d3","SigW_Maha_d4",
]
corruptions = ["full_shuffle","block_shuffle","sign_shuffle","time_reverse"]

A = np.array([[R["auc_paired"][m][c] for c in corruptions] for m in metrics_order])
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(A, cmap="RdBu_r", vmin=0.42, vmax=0.70, aspect="auto")
ax.set_xticks(range(len(corruptions))); ax.set_xticklabels(corruptions, rotation=30, ha="right")
ax.set_yticks(range(len(metrics_order))); ax.set_yticklabels(metrics_order)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j, i, f"{A[i,j]:.3f}", ha="center", va="center",
                color="white" if abs(A[i,j]-0.5)>0.05 else "black", fontsize=8)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Paired AUC (real vs corrupted)")
# add group separators
for ypos in [5.5, 7.5, 10.5]:
    ax.axhline(ypos, color="black", linewidth=0.8)
ax.set_title("Paired AUC of single-path metrics vs corruptions\nS&P 500 daily returns, W=128, n=2090 test windows")
fig.tight_layout()
fig.savefig("fig1_auc_heatmap.png", dpi=150)
print("Wrote fig1_auc_heatmap.png")

# fig2: bootstrap relative improvement bar plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, key, title in zip(axes, ["all4","marginal_blind"],
        ["All 4 corruptions\n(best marginal: MMD)",
         "Marginal-blind subset\n(full, block, time-reverse)"]):
    boot = R[f"bootstrap_{key}"]
    cands = [k for k in boot if k.startswith("SigW")]
    means = [boot[c]["rel_pct_mean"] for c in cands]
    los   = [boot[c]["rel_pct_lo"]   for c in cands]
    his   = [boot[c]["rel_pct_hi"]   for c in cands]
    y = np.arange(len(cands))
    ax.errorbar(means, y, xerr=[np.array(means)-np.array(los),
                                 np.array(his)-np.array(means)],
                fmt="o", color="C0", capsize=4)
    ax.set_yticks(y); ax.set_yticklabels(cands)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(3, color="red", linewidth=0.8, linestyle="--",
               label="3% acceptance threshold")
    ax.set_xlabel("Relative AUC improvement (%)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
fig.suptitle("Sig-Wasserstein vs strongest marginal-only metric — bootstrap 95% CI")
fig.tight_layout()
fig.savefig("fig2_bootstrap.png", dpi=150)
print("Wrote fig2_bootstrap.png")
