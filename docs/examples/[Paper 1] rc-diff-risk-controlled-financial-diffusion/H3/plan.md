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
