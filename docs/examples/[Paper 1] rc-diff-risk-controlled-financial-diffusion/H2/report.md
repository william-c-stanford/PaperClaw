## Verdict
SUPPORTED

## Discussion
The H2 experiment supports the claim that explicit volatility conditioning can produce monotonic realized-volatility control in generated financial return paths. The proposed conditioned diffusion model achieved Spearman(requested volatility, realized volatility) = 0.9788 with a 0.0 monotonicity violation rate, meeting the plan's primary acceptance gate against the best controllable baseline, Sig-W-GAN-lite, whose Spearman was only 0.0171 with a 0.5 violation rate. The reported improvement over Sig-W-GAN-lite was +0.9617 absolute Spearman with 95% CI [0.9381, 0.9819], comfortably above the required 3% relative improvement threshold.

A subtle but important caveat is that unconditional/post-hoc-binned baselines also show near-perfect Spearman rank correlation: TimeGAN, C-RNN-GAN, and the unconditional diffusion all report Spearman ≈ 0.9798 with 0.0 violation rate. This means rank monotonicity alone is not a sufficient demonstration of useful risk control, because post-hoc sorting or binning can create monotonic rank relationships without producing calibrated stress scenarios. The stronger evidence for explicit conditioning is its absolute risk calibration: diff_full has much lower VaR calibration error (0.1454) and ES calibration error (0.0499) than unconditional diffusion (0.3747 and 0.4145), C-RNN-GAN (0.5823 and 0.6046), TimeGAN (10.5525 and 7.7718), and Sig-W-GAN-lite (0.9043 and 0.9162).

The shuffled-label ablation is especially diagnostic. When volatility labels are randomly permuted, Spearman collapses to -0.0056 and the violation rate rises to 0.4583, showing that the model uses the conditioning signal rather than merely reproducing a generic unconditional volatility mixture. The stylized-fact auxiliary loss is useful but not uniformly dominant: it improves VaR calibration (0.1454 vs 0.1748 for no-SF) and ES calibration (0.0499 vs 0.0685), while slightly worsening max-drawdown calibration (0.2069 vs 0.1821), lag-5 absolute-autocorrelation error (0.0341 vs 0.0255), signature proxy distance (0.00663 vs 0.00626), and tail-index error (0.0969 vs 0.0716). This suggests the risk-control interface works, but the auxiliary losses need careful weighting so they improve risk calibration without trading away other stylized facts.

## Key Findings
1. Useful: explicit volatility conditioning is effective against the closest controllable baseline, with diff_full Spearman = 0.9788 versus Sig-W-GAN-lite = 0.0171 and a reported +0.9617 absolute improvement with 95% CI [0.9381, 0.9819].
2. Useful: the learned conditioning channel is real, because shuffled volatility labels collapse to Spearman = -0.0056 and monotonicity violation rate = 0.4583, while diff_full maintains Spearman = 0.9788 and violation rate = 0.0.
3. Useful: explicit conditioning gives substantially better absolute risk calibration than post-hoc-binned unconditional generation, with diff_full VaR/ES errors = 0.1454/0.0499 versus unconditional diffusion = 0.3747/0.4145, C-RNN-GAN = 0.5823/0.6046, and TimeGAN = 10.5525/7.7718.
4. Not useful alone: Spearman monotonicity is too easy to satisfy with post-hoc binning, since unconditional diffusion, TimeGAN, and C-RNN-GAN all reach Spearman ≈ 0.9798 despite much worse VaR, ES, max-drawdown, autocorrelation, and/or signature-proxy errors.
5. Interesting: the stylized-fact loss improves tail-risk calibration but creates mixed trade-offs, improving VaR error from 0.1748 to 0.1454 and ES error from 0.0685 to 0.0499, while worsening max-drawdown calibration from 0.1821 to 0.2069 and tail-index error from 0.0716 to 0.0969.

## Future Directions
- Replace or supplement Spearman rank correlation with calibration-aware control metrics, such as target-realized slope, absolute target error per bin, expected calibration error over volatility quantiles, and conditional VaR/ES calibration.
- Test control generalization on regime-shift splits, especially pre-2020 training with 2020–2024 stress-period holdout, to verify the conditioning interface outside ordinary volatility regimes.
- Tune or learn the stylized-fact loss weights, because the current auxiliary loss improves VaR/ES but worsens drawdown and tail-index fidelity.
- Extend H2 from volatility-only control to joint volatility + drawdown + tail-severity control, checking whether monotonicity and calibration survive multi-objective conditioning.

## Proposed Hypotheses
- H2.1 (sub of H2): Post-hoc volatility binning can create high Spearman monotonicity without calibrated risk control. Test/criterion: post-hoc-binned unconditional models match Spearman within 1% of controlled diffusion but have at least 2× worse VaR or ES calibration error.
- H2.2 (sub of H2): Stylized-fact auxiliary losses improve tail-risk calibration but introduce trade-offs with drawdown and tail-index fidelity. Test/criterion: tuned loss variants reduce VaR/ES error by at least 10% without increasing max-drawdown or tail-index error by more than 5%.
- H2.3 (sub of H2): Volatility conditioning remains calibrated under regime-shift evaluation. Test/criterion: a model trained pre-2020 preserves monotonicity violation rate below 5% and volatility calibration error degradation below 25% on 2020–2024 holdout windows.
