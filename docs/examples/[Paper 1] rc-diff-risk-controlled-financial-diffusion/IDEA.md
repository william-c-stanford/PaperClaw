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
RC-Diff — Two-word abbreviation for the proposed risk-controlled diffusion model that injects a volatility/risk embedding into each reverse-denoising block.
Stylized-fact loss — A training penalty measuring mismatch in known empirical properties of asset returns.
Path-level financial fidelity — Realism of entire return trajectories, including ordering and temporal dependence, not only pointwise distributions.

## Open Questions
How can risk controls avoid generating arbitrage-like artifacts? Which stylized facts should be enforced during training versus reserved for evaluation? Can controls generalize across market regimes without leaking future information?