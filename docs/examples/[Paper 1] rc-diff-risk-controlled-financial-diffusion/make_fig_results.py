import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05))
fig.subplots_adjust(left=0.07, right=0.995, bottom=0.30, top=0.82, wspace=0.42)

# Panel 1: risk calibration
labels = ["Unconditional\ndiffusion", "RC-Diff\nwithout\nstylized loss", "RC-Diff"]
var_error = np.array([0.3747, 0.1748, 0.1454])
es_error = np.array([0.4145, 0.0685, 0.0499])
x = np.arange(len(labels))
width = 0.34
axes[0].bar(x - width/2, var_error, width, label="VaR error", color="#6B8FBF")
axes[0].bar(x + width/2, es_error, width, label="ES error", color="#D08C60")
axes[0].set_title("Risk calibration")
axes[0].set_ylabel("Calibration error")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].legend(frameon=False, loc="upper right")
axes[0].spines[["top", "right"]].set_visible(False)
axes[0].grid(axis="y", alpha=0.25, linewidth=0.5)

# Panel 2: temporal audit
metrics = ["Gaussian\nMMD", "Signature\nscore\ndepth three"]
auc = [0.5404, 0.5657]
colors = ["#8FAADC", "#4F7CAC"]
axes[1].bar(np.arange(2), auc, color=colors, width=0.55)
axes[1].axhline(0.5, color="#555555", linewidth=0.8, linestyle="--")
axes[1].set_title("Temporal audit")
axes[1].set_ylabel("Detection AUC")
axes[1].set_ylim(0.48, 0.58)
axes[1].set_xticks(np.arange(2))
axes[1].set_xticklabels(metrics)
axes[1].spines[["top", "right"]].set_visible(False)
axes[1].grid(axis="y", alpha=0.25, linewidth=0.5)
axes[1].text(1, 0.568, "Best", ha="center", va="bottom", fontsize=7)

# Panel 3: auxiliary loss ablation
tail_labels = ["RC-Diff\nwithout\nstylized loss", "RC-Diff\nwith\nstylized loss"]
tail_error = [1.4484, 2.3961]
axes[2].bar(np.arange(2), tail_error, color=["#79A77E", "#C66B6B"], width=0.55)
axes[2].set_title("Auxiliary loss ablation")
axes[2].set_ylabel("Tail error")
axes[2].set_xticks(np.arange(2))
axes[2].set_xticklabels(tail_labels)
axes[2].spines[["top", "right"]].set_visible(False)
axes[2].grid(axis="y", alpha=0.25, linewidth=0.5)
axes[2].annotate("Worse", xy=(1, 2.3961), xytext=(1, 2.62), ha="center", va="bottom", fontsize=7,
                 arrowprops=dict(arrowstyle="-|>", linewidth=0.6, color="#555555"))

fig.suptitle("Key empirical results for RC-Diff", y=0.98, fontsize=10)
fig.savefig("/fig_results.pdf", bbox_inches="tight")
fig.savefig("/fig_results.png", dpi=220, bbox_inches="tight")
print("wrote /fig_results.pdf and /fig_results.png")
