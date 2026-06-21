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
Signature-based path metrics identify temporal-dependence failures in generated financial time series that are not detected by marginal return-distribution metrics.

## Datasets
S&P 500 constituents daily OHLCV from Yahoo Finance, using adjusted-close log-return windows from roughly 2010–2024. Use approximately 350–500 liquid tickers after survivorship/missing-data filtering, 3,000–3,700 trading days per ticker, and rolling windows of 64 or 128 days for train/validation/test splits by time. Include a synthetic diagnostic suite derived from held-out real windows: shuffled-return paths, block-shuffled paths, GARCH-fitted resamples, correlation-scrambled multi-asset paths, and model-generated paths from the main experiment.

## Baselines / Ablations
Compare the proposed diffusion model against TimeGAN, C-RNN-GAN, Sig-Wasserstein GAN, CoFinDiff if implementation details are available, and an ablated diffusion model without stylized-fact losses or risk controls. For metric-specific ablations, include marginal-only evaluation using Kolmogorov-Smirnov distance, Wasserstein distance on returns, tail-index error, VaR/ES error, and histogram/MMD-style distribution metrics; temporal metrics using absolute-return autocorrelation error and correlation-matrix distance; and signature-based evaluation using Sig-Wasserstein distance at multiple signature truncation depths. Include controlled path-destruction ablations that preserve marginal returns while destroying ordering: full shuffle, block shuffle, sign shuffle, and within-window time reversal.

## Metrics
Primary metric: temporal-failure detection AUC, where each metric ranks real held-out paths above corrupted paths that preserve marginal return distributions but break temporal dependence. Secondary metrics: Kendall/Spearman correlation between each metric and corruption severity, false-negative rate at fixed 5% false-positive rate, agreement with downstream forecasting/stress-test degradation using Nixtla/neuralforecast and thuml/Time-Series-Library models, sensitivity by signature truncation depth, and runtime per 1,000 generated windows.

## Acceptance Criteria
SUPPORTED if Sig-Wasserstein distance achieves at least a 3% relative improvement in mean temporal-failure detection AUC over the strongest marginal-only metric, averaged across shuffle, block-shuffle, sign-shuffle, and time-reversal corruptions with 95% bootstrap confidence interval excluding zero. The criterion does not require Sig-Wasserstein to beat every temporal baseline on every corruption type, but it must consistently flag failures that marginal return-distribution metrics miss.

## Resource Estimation
Expected compute: one modest GPU with 8–16 GB VRAM for generating or loading model samples and running downstream neural forecasting probes; CPU-only is acceptable for the metric-only corruption study but slower. Runtime: approximately 2–4 hours for constructing windows and corruption sets, 1–3 hours for metric computation over 10,000–50,000 windows depending on signature depth, and 4–8 additional GPU hours if downstream forecasting probes are run. Disk: 5–20 GB for raw Yahoo Finance OHLCV, processed windows, generated samples, corruption sets, and cached metric outputs.

## Feasibility
FEASIBLE — no hardware file was detected, but the core H3 test can run on a modest single-GPU/CPU machine because it mainly requires offline metric computation over fixed real, corrupted, and generated return windows.


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
