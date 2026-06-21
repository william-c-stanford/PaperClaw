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