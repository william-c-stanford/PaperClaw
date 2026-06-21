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
Conditioning a financial diffusion model on target volatility produces generated return paths whose realized volatility is monotonic in the requested volatility level.

## Datasets
- S&P 500 constituents daily OHLCV from Yahoo Finance, approximately 450–500 liquid equities over 10–20 years after survivorship/data-quality filtering, converted to rolling log-return windows of 64–252 trading days.
- SPY ETF daily OHLCV from Yahoo Finance over the same period as a market-index sanity benchmark for single-asset volatility control.
- Optional robustness split: pre-2020 train/validation and 2020–2024 stress/regime-shift holdout to test calibration outside calm regimes.

## Baselines / Ablations
- TimeGAN: sequence GAN baseline for financial time-series generation; evaluate whether post-hoc volatility binning can match explicit conditioning.
- C-RNN-GAN: recurrent adversarial generator baseline for temporal financial sequences.
- Sig-Wasserstein GAN: closest path-signature competitor; compare path fidelity and volatility controllability where conditioning is available or approximated by volatility-stratified training.
- CoFinDiff: closest controllable financial diffusion baseline, if implementation/details are available; compare requested-control fidelity directly.
- Proposed controlled diffusion: diffusion model conditioned on target realized-volatility level and, in the broader experiment, drawdown targets.
- Ablated diffusion without risk controls: same architecture and training data but no volatility/drawdown conditioning.
- Ablated diffusion with shuffled or noisy volatility labels: isolates whether monotonicity comes from learned conditioning rather than unconditional regime mixture.
- Ablated diffusion without stylized-fact losses: tests whether explicit risk control remains calibrated without auxiliary financial regularization.

## Metrics
- Primary: Spearman rank correlation between requested volatility quantile and realized annualized volatility of generated samples, aggregated across assets and rolling-window lengths.
- Secondary: volatility calibration error between target and generated realized volatility, monotonicity violation rate across adjacent target bins, VaR calibration error, expected shortfall calibration error, max-drawdown calibration error, correlation-matrix distance, absolute-return autocorrelation error, and Sig-Wasserstein distance to matched real windows.
- Sanity checks: unconditional return mean near zero after demeaning, no systematic price-path arbitrage artifacts, and stable results across random seeds.

## Acceptance Criteria
SUPPORTED if the proposed controlled diffusion improves the primary metric, Spearman rank correlation between requested volatility bin and generated realized volatility, by at least 3% relative over the best controllable baseline or the ablated diffusion without risk controls, averaged across three random seeds and both S&P 500 constituent and SPY benchmark evaluations, with bootstrap 95% confidence intervals excluding zero improvement.

## Resource Estimation
- Compute: 1 GPU with 12–24 GB VRAM is sufficient for initial experiments using 64–128 day windows, batch sizes of 64–256, and a compact 1D U-Net/Transformer diffusion backbone; 24–48 GB VRAM is preferred for multi-asset conditioning and larger windows.
- GPUs: 1 GPU for pilot and main single-asset/marginal experiments; 2–4 GPUs helpful but not required for larger multi-asset correlation experiments.
- Runtime: approximately 8–24 GPU-hours per model variant for pilot-scale daily-return windows; 2–5 days total for baselines, ablations, three seeds, and evaluation on a single modern GPU.
- Dataset/disk: raw Yahoo Finance OHLCV under 5 GB; processed rolling-window tensors, generated samples, checkpoints, and metric artifacts approximately 20–80 GB depending on sample count and retained checkpoints.

## Feasibility
FEASIBLE — no `HARDWARE.md` was detected, but the plan fits a modest single-GPU/CPU machine if run with compact daily-return windows, limited seeds, and staged baseline training.

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
