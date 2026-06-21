## Hypothesis
A diffusion model trained with differentiable stylized-fact losses reduces tail-index error in generated financial returns compared with the same diffusion model without those losses.

## Datasets
- **S&P 500 constituent OHLCV from Yahoo Finance**: daily data for roughly 400–500 liquid names over about 10–20 years, converted to rolling return windows.
- **Optional market proxy series**: SPY / sector ETFs as additional held-out sanity checks for cross-asset behavior.
- Windowing: 64–256 day return sequences, with train/validation/test splits by time to avoid leakage.

## Baselines / Ablations
- **Diffusion without stylized-fact losses**: the direct ablation needed to isolate the effect of the new training objective.
- **TimeGAN**: strongest common sequential GAN baseline for financial time series realism.
- **C-RNN-GAN**: older recurrent adversarial generator baseline for sequence synthesis.
- **Sig-Wasserstein GAN**: closest path-signature competitor and strongest baseline for preserving temporal structure.
- **CoFinDiff**: closest controllable financial diffusion baseline if reproducible details are available.
- Additional ablations: remove tail-loss only, remove volatility-clustering loss only, and remove cross-asset dependence loss only.

## Metrics
- **Primary**: absolute tail-index error on generated vs. real return windows, averaged across assets and test splits.
- **Secondary**: error in autocorrelation of absolute returns, correlation-matrix distance, and Sig-Wasserstein distance.
- Reporting: mean ± standard error over multiple random seeds and evaluation windows.

## Acceptance Criteria
- Supported if the full model achieves a consistent **2–5% relative reduction** in **mean absolute tail-index error** versus the no-stylized-fact ablation, aggregated across assets and seeds, with a paired significance test indicating the improvement is reliable.

## Resource Estimation
- **Compute**: 1 GPU with about **16–24 GB VRAM** is sufficient for sequence diffusion training at moderate window sizes.
- **Runtime**: roughly **8–24 hours** for one full training run plus evaluation, depending on window length and asset count.
- **Storage**: about **1–5 GB** for cached OHLCV data, features, checkpoints, and generated samples.
- **Runs**: 3–5 seeds for the main comparison, plus short ablation runs.

## Feasibility
FEASIBLE — the experiment fits a modest single-GPU machine with standard CPU preprocessing, and the main comparison is against a direct diffusion ablation rather than a large-scale multimodal setup.