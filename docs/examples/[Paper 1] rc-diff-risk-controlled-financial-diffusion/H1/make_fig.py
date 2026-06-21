import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "results.json")) as f:
    R = json.load(f)
m = R["experiments"][0]["metrics"]

methods = ["DiffNoStyl", "DiffStyl", "SeqGAN"]
labels  = ["Diffusion\n(no styl loss)", "Diffusion\n(+styl losses)", "SeqGAN\n(GRU GAN)"]
metrics = ["tail_index_err", "acf_abs_err", "corr_matrix_dist", "sig_wasserstein"]
titles  = ["Tail-index error\n(Hill, lower=better)",
           "ACF-of-|r| error\n(lower=better)",
           "Correlation matrix Frobenius dist\n(lower=better)",
           "Truncated Sig-Wasserstein dist\n(lower=better)"]

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
colors = ["#1f77b4", "#d62728", "#7f7f7f"]
for ax, metric, title in zip(axes, metrics, titles):
    vals = [m[meth][metric] for meth in methods]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_title(title, fontsize=10)
    if metric == "tail_index_err":
        # log scale because SeqGAN dwarfs
        ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
plt.suptitle("H1 ablation: differentiable stylized-fact losses on a 1D-UNet DDPM (S&P 500, 3 seeds)", fontsize=11)
plt.tight_layout()
out = os.path.join(HERE, "fig1.png")
plt.savefig(out, dpi=110, bbox_inches="tight")
print("Wrote", out)
