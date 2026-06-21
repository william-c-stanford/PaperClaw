You are running headlessly as an autonomous coding agent in the CURRENT working
directory to execute a research experiment end-to-end. You have a Linux shell,
Python, and full read/write access to this directory. Work autonomously to
completion — do NOT ask questions or wait for confirmation.

# Research idea
IDEA.md:
# Stylized-Fact Preserving Financial Diffusion with Distributional Risk Controls

## Domain & Literature (domain pin)
Time Series Generation + Generative Modeling; builds on Mogren (2016) C-RNN-GAN, Yoon et al. (2019) TimeGAN, Shujian Liao et al. (2023) Sig-Wasserstein GANs for conditional time series generation, Yuki Tanaka et al. (2025) CoFinDiff: Controllable Financial Diffusion Model for Time Series Generation, Ho et al. (2020) Denoising Diffusion Probabilistic Models, and Song et al. (2021) Score-Based Generative Modeling through Stochastic Differential Equations.

## Target Venue
NeurIPS — Conference on Neural Information Processing Systems

## Keywords
financial time series, stylized facts, risk control, signature metrics, diffusion, stress testing

## Background
Financial generators must preserve heavy tails, volatility clustering, leverage effects, cross-asset dependence, and realistic drawdowns. Sig-Wasserstein GANs introduce path-signature-based objectives for conditional financial generation, while controllable financial diffusion points toward richer scenario control.

## Research Gap
Many financial generators optimize marginal realism or forecasting utility but fail to expose explicit controls for risk-relevant distributional properties such as tail heaviness, volatility regime, correlation breakdown, and maximum drawdown. If controls are not faithful, synthetic scenarios can mislead stress testing and portfolio risk assessment. The gap matters because financial synthetic data is often used precisely where rare but plausible extremes are important.

## Motivation
A diffusion model with auditable controls over stylized facts could generate realistic stress scenarios for model validation, risk analysis, and robust trading-system evaluation.

## Main Result
Baselines: compare against TimeGAN (sequence GAN baseline for financial time series), C-RNN-GAN (recurrent adversarial generation baseline), Sig-Wasserstein GAN (closest path-signature competitor), CoFinDiff (closest controllable financial diffusion baseline, if implementation/details are available), and an ablated diffusion model without stylized-fact losses or risk controls. Main experiment: train on daily S&P 500 constituents / Yahoo Finance OHLCV return windows, condition on volatility and drawdown targets, and evaluate generated paths against held-out real windows using tail-index error, absolute-return autocorrelation error, correlation-matrix distance, VaR/ES/max-drawdown calibration error, Sig-Wasserstein distance, and downstream stress-test utility. Target outcome: the proposed model should match or improve path-level fidelity while providing monotonic, calibrated risk controls that baselines lack.

## Root Hypotheses
H1: Adding differentiable stylized-fact losses improves preservation of heavy tails, volatility clustering, and cross-asset dependence; test on S&P 500 / Yahoo Finance OHLCV using tail-index error, autocorrelation of absolute returns, and correlation-matrix distance.
H2: Distributional risk controls produce monotonic and calibrated changes in generated drawdown and volatility regimes; test conditional samples from S&P 500 / Yahoo Finance OHLCV for realized volatility, VaR, expected shortfall, and max drawdown calibration.
H3: Signature-based evaluation detects path-level failures missed by marginal metrics; compare Sig-Wasserstein, TSGBench-style metrics, and downstream forecasting utility with Nixtla/neuralforecast and thuml/Time-Series-Library.

## Current Findings
_None yet._

## Key Concepts
Distributional risk control — A conditioning interface that targets risk properties such as volatility, tail severity, correlation regime, and drawdown.
Stylized-fact loss — A training penalty measuring mismatch in known empirical properties of asset returns.
Path-level financial fidelity — Realism of entire return trajectories, including ordering and temporal dependence, not only pointwise distributions.

## Open Questions
How can risk controls avoid generating arbitrage-like artifacts? Which stylized facts should be enforced during training versus reserved for evaluation? Can controls generalize across market regimes without leaking future information?

# Experiment plan (FOLLOW IT EXACTLY)
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

# Your task
1. Set up the environment (install missing packages with `pip install -q ...`).
2. Use the EXACT datasets, baselines/ablations, and metrics the plan names. Load
   the REAL named datasets (download via `datasets`/torchvision/sklearn/openml/a
   public URL). SUBSAMPLE for speed and use the GPU if available — but NEVER
   substitute synthetic data for a named benchmark. If a dataset is truly
   unobtainable, use the closest REAL alternative and say why in "data_note".
3. Write your script(s), run them, inspect the output, and fix errors until the
   experiment completes with REAL measured numbers (honest — including negative
   or inconclusive outcomes; never fabricate).
4. Optionally save conceptual/result figures as PNG files in this directory.

# Required output
Write a file named `results.json` in THIS directory with EXACTLY this schema:
{
  "experiments": [
    {"name": str, "setup": str,
     "metrics": {"<method or baseline>": {"<metric>": number, ...}, ...},
     "hypothesis": str,
     "verdict": "SUPPORTED" | "REFUTED" | "INCONCLUSIVE",
     "status": "POSITIVE" | "MIXED" | "NEGATIVE",
     "observations": str}
  ],
  "summary": str,
  "data_note": str,   // OPTIONAL: only if a planned dataset could not be used
  "figures": [{"file": "fig1.png", "caption": str}]   // optional; omit if none
}
`results.json` is the deliverable — the run is judged complete only once it exists.
