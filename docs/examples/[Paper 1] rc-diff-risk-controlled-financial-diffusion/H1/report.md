## Verdict
REFUTED

## Discussion
H1 is refuted against the preregistered acceptance criterion. The plan required a consistent 2–5% relative reduction in mean absolute tail-index error versus the no-stylized-fact diffusion ablation, supported by a paired significance test. Instead, the stylized-loss model worsened the primary metric substantially: DiffStyl reached a mean Hill tail-index error of 2.396, compared with 1.448 for DiffNoStyl, a 65.4% relative increase rather than a reduction. The one-sided paired Wilcoxon test for the planned improvement gave p=1.0, so there is no evidence for the hypothesized gain.

The negative result is not just a narrow failure of the primary metric. Two path-level secondary metrics also deteriorated: correlation-matrix distance increased from 1.710 to 5.555, and truncated Sig-Wasserstein distance increased from 0.001056 to 0.018769. The only metric that did not materially worsen was absolute-return autocorrelation error, which was nearly flat at 0.03196 for DiffNoStyl versus 0.03126 for DiffStyl. This suggests the auxiliary losses did not generally improve stylized-fact fidelity and may have disrupted the learned score in ways that damage tail and dependence structure.

The most plausible interpretation is that the operationalization of H1 was too naive: the differentiable stylized-fact losses were applied to x0_hat during training for diffusion steps with t < 0.6*T_diff. At nontrivial noise levels, x0_hat can be a biased and low-SNR estimate of the clean return path, so matching empirical Hill tail indices, absolute-return ACF, and correlation matrices on x0_hat can produce misleading gradients. Those gradients may pull the denoiser away from the correct epsilon-prediction objective, explaining why DiffStyl underperformed the direct diffusion ablation.

A useful boundary condition also emerged. SeqGAN performed dramatically worse than both diffusion variants on the primary and secondary metrics, with tail-index error of 92.607, absolute-return ACF error of 0.4509, correlation distance of 10.355, and Sig-Wasserstein distance of 0.01163. This supports keeping the diffusion ablation as the main comparator for future H1-style tests, while treating the sequence-GAN baseline as a weaker reference rather than the core competitor.

## Key Findings
1. Useful: The no-stylized-loss diffusion baseline is strong and should remain the central ablation comparator, with tail-index error 1.448 versus 92.607 for SeqGAN and much lower ACF/correlation errors than SeqGAN.
2. Not useful: Applying Hill tail-index, |r|-ACF, and cross-asset correlation losses directly to x0_hat did not improve the primary objective; DiffStyl increased tail-index error from 1.448 to 2.396, a 65.4% relative worsening, with Wilcoxon p=1.0 for the planned one-sided improvement.
3. Not useful: The combined stylized-fact loss damaged dependence/path fidelity, increasing correlation-matrix distance from 1.710 to 5.555 and Sig-Wasserstein distance from 0.001056 to 0.018769.
4. Useful but limited: The |r|-ACF component appears comparatively harmless or weakly helpful, with absolute-return autocorrelation error nearly unchanged/slightly lower at 0.03126 for DiffStyl versus 0.03196 for DiffNoStyl.
5. Interesting: The result points to a training-signal problem rather than a failure of the stylized-fact targets themselves, because losses computed on noisy x0_hat estimates likely produce biased moment-matching gradients at moderate diffusion noise.

## Future Directions
- Retest stylized-fact losses only at very low diffusion timesteps, where x0_hat is closer to the clean return path and moment statistics are less biased.
- Move stylized-fact objectives from noisy training reconstructions to actual generated samples, using short DDIM rollouts or periodic sample-level fine-tuning.
- Isolate each auxiliary loss instead of using the combined objective, especially separating the apparently less harmful |r|-ACF loss from tail-index and correlation losses.
- Add gradient diagnostics that compare auxiliary-loss gradient direction with epsilon-MSE gradient direction across diffusion timesteps.
- Expand the asset universe after the mechanism is fixed, since this run used 16 liquid S&P 500 names rather than the planned 400–500 constituents.

## Proposed Hypotheses
1. (sub of H1) Restricting stylized-fact losses to very low diffusion timesteps improves tail-index error without damaging correlation fidelity. Test: compare low-t-only DiffStyl against DiffNoStyl and require at least 2% tail-index error reduction with no increase in correlation-matrix distance.
2. (sub of H1) Computing stylized-fact losses on short DDIM-generated sample paths is more reliable than computing them on noisy x0_hat estimates. Test: compare sample-level DiffStyl against x0_hat DiffStyl and DiffNoStyl, requiring lower tail-index error than both.
3. (sub of H1) The absolute-return ACF loss is safer than tail-index and correlation losses when used as an auxiliary diffusion objective. Test: run single-loss ablations and require ACF error reduction without more than 1% degradation in tail-index error.
