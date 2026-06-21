"""
H3 v2: Refined version of the signature-based detection metric.

Changes from v1
---------------
1. Replace L2-distance-to-mean signature with a properly normalised
   *Mahalanobis* distance in signature space.  This is the witness-function
   form of a Sig-Wasserstein (signature-MMD) two-sample test applied per
   path, and is the standard way to convert a distributional signature
   metric into a per-path detector.
2. Add a per-path absolute-return autocorrelation discrepancy that
   compares the path's empirical ACF profile to the reference mean ACF
   under the reference covariance.
3. Compute paired AUC (sgn(s_corrupt - s_real) averaged over paths) and
   matched bootstrap CI on the relative improvement vs the strongest
   marginal-only metric.
4. Add a control: 'time_reverse' is sometimes invariant under
   stylised-fact metrics, so we still report it for completeness.

The reference distribution and corruptions are the same as v1.
"""
import os, sys, time, json, math, warnings
import numpy as np
import pandas as pd
from scipy import stats
import iisignature
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- load
print("Loading prices.pkl ...")
close = pd.read_pickle("prices.pkl").sort_index()
log_returns = np.log(close).diff().dropna(how="all").fillna(0.0)

split_date = pd.Timestamp("2019-01-01")
ref_returns_panel = log_returns.loc[:split_date]
test_returns_panel = log_returns.loc[split_date:]
print("ref panel:", ref_returns_panel.shape, "test panel:", test_returns_panel.shape)

W = 64
STRIDE = 32

def windows_from_panel(panel, w=W, stride=STRIDE):
    out = []
    arr = panel.values
    for j in range(arr.shape[1]):
        col = arr[:, j]
        for i in range(0, len(col) - w + 1, stride):
            seg = col[i:i+w]
            if np.all(np.isfinite(seg)) and seg.std() > 1e-8:
                out.append(seg.astype(np.float64))
    return np.stack(out, axis=0)

ref_windows = windows_from_panel(ref_returns_panel)
test_windows = windows_from_panel(test_returns_panel)
print(f"Reference windows: {ref_windows.shape}")
print(f"Test windows:      {test_windows.shape}")

MAX_REF = 4000
if len(ref_windows) > MAX_REF:
    idx = rng.choice(len(ref_windows), size=MAX_REF, replace=False)
    ref_windows = ref_windows[idx]
MAX_TEST = 4000
if len(test_windows) > MAX_TEST:
    idx = rng.choice(len(test_windows), size=MAX_TEST, replace=False)
    test_windows = test_windows[idx]
print(f"Ref windows used: {len(ref_windows)}")
print(f"Test windows used: {len(test_windows)}")

ref_returns_pool = ref_windows.reshape(-1)

# ---------------------------------------------------------------- corruptions
def corrupt_full_shuffle(x, rng):
    y = x.copy(); rng.shuffle(y); return y
def corrupt_block_shuffle(x, rng, b=8):
    n = len(x); nb = n // b
    blocks = [x[i*b:(i+1)*b] for i in range(nb)]
    rng.shuffle(blocks)
    extra = [x[nb*b:]] if n % b else []
    return np.concatenate(blocks + extra)
def corrupt_sign_shuffle(x, rng):
    s = rng.choice([-1.0, 1.0], size=x.shape)
    return x * s
def corrupt_time_reverse(x, rng=None):
    return x[::-1].copy()

CORRUPTIONS = {
    "full_shuffle":  corrupt_full_shuffle,
    "block_shuffle": corrupt_block_shuffle,
    "sign_shuffle":  corrupt_sign_shuffle,
    "time_reverse":  corrupt_time_reverse,
}

# ---------------------------------------------------------------- marginal/temporal metrics
ref_sorted = np.sort(ref_returns_pool)
ref_mean = float(ref_returns_pool.mean())
ref_std  = float(ref_returns_pool.std())

def metric_ks(x, ref_sorted=ref_sorted):
    return stats.ks_2samp(x, ref_sorted, method="asymp").statistic

ref_sample_for_w = rng.choice(ref_returns_pool, size=2000, replace=False)
def metric_w1(x, ref=ref_sample_for_w):
    return stats.wasserstein_distance(x, ref)

def hill_tail(absr, k_frac=0.10):
    n = len(absr); k = max(8, int(n * k_frac))
    sr = np.sort(absr)[::-1]
    top = sr[:k]; thr = sr[k] if k < n else sr[-1]
    thr = max(thr, 1e-12)
    return float(np.mean(np.log(np.maximum(top, thr*1e-6) / thr)))
ref_hill = hill_tail(np.abs(ref_returns_pool))
def metric_tail(x):
    return abs(hill_tail(np.abs(x)) - ref_hill)

ref_var5 = -np.quantile(ref_returns_pool, 0.05)
def metric_var(x):
    return abs(-np.quantile(x, 0.05) - ref_var5)

mask = ref_returns_pool <= -ref_var5
ref_es5 = -ref_returns_pool[mask].mean()
def metric_es(x):
    v5 = -np.quantile(x, 0.05)
    t = x[x <= -v5]
    if len(t) == 0: return abs(ref_es5)
    return abs(-t.mean() - ref_es5)

mmd_ref = rng.choice(ref_returns_pool, size=512, replace=False)
mmd_sigma = max(np.median(np.abs(mmd_ref - np.median(mmd_ref))), 1e-6)
def gauss_k(a, b, sigma):
    return np.exp(-((a[:, None] - b[None, :]) ** 2) / (2 * sigma * sigma))
Kyy = gauss_k(mmd_ref, mmd_ref, mmd_sigma).mean()
def metric_mmd(x, ref=mmd_ref, s=mmd_sigma, kyy=Kyy):
    return float(max(gauss_k(x, x, s).mean() + kyy - 2*gauss_k(x, ref, s).mean(), 0.0))

LAGS = [1, 2, 3, 5, 10]
def acf(x, lag):
    x = x - x.mean()
    denom = (x*x).sum()
    if denom < 1e-20: return 0.0
    return float((x[:-lag] * x[lag:]).sum() / denom)

# Reference abs/sign acf VECTORS (per-path) so we can compute Mahalanobis
ref_abs_acf_vecs = np.array([[acf(np.abs(w), L) for L in LAGS] for w in ref_windows])
ref_sgn_acf_vecs = np.array([[acf(np.sign(w), L) for L in LAGS] for w in ref_windows])

abs_acf_mean = ref_abs_acf_vecs.mean(0)
sgn_acf_mean = ref_sgn_acf_vecs.mean(0)
abs_acf_invcov = np.linalg.pinv(np.cov(ref_abs_acf_vecs, rowvar=False) + 1e-6 * np.eye(len(LAGS)))
sgn_acf_invcov = np.linalg.pinv(np.cov(ref_sgn_acf_vecs, rowvar=False) + 1e-6 * np.eye(len(LAGS)))

def metric_abs_acf(x):
    v = np.array([acf(np.abs(x), L) for L in LAGS]) - abs_acf_mean
    return float(v @ abs_acf_invcov @ v)
def metric_sgn_acf(x):
    v = np.array([acf(np.sign(x), L) for L in LAGS]) - sgn_acf_mean
    return float(v @ sgn_acf_invcov @ v)

# ---------------------------------------------------------------- signature metric (Mahalanobis)
TIME = np.linspace(0.0, 1.0, W)
def path_with_time(x, scale):
    return np.stack([TIME, np.cumsum(x)/scale], axis=1)

norm_scale = ref_std * math.sqrt(W) + 1e-12

SIG_DEPTHS = [2, 3, 4]
sig_dim = {d: len(iisignature.sig(np.zeros((W,2)), d)) for d in SIG_DEPTHS}
print("Signature dimensions:", sig_dim)

# Precompute reference signatures
def compute_sig(x, depth):
    return np.asarray(iisignature.sig(path_with_time(x, norm_scale), depth), dtype=np.float64)

print("Building reference signatures ...")
ref_sigs = {}
ref_sig_mean = {}
ref_sig_invcov = {}
for d in SIG_DEPTHS:
    feats = np.stack([compute_sig(w, d) for w in ref_windows])
    ref_sigs[d] = feats
    ref_sig_mean[d] = feats.mean(0)
    cov = np.cov(feats, rowvar=False) + 1e-8 * np.eye(feats.shape[1])
    ref_sig_invcov[d] = np.linalg.pinv(cov)
    print(f"  depth {d}: features {feats.shape[1]}")

def metric_sigw_L2(x, d):
    return float(np.linalg.norm(compute_sig(x, d) - ref_sig_mean[d]))
def metric_sigw_Maha(x, d):
    v = compute_sig(x, d) - ref_sig_mean[d]
    return float(v @ ref_sig_invcov[d] @ v)

# ---------------------------------------------------------------- metrics dict
METRICS = {
    "KS":            metric_ks,
    "Wasserstein1":  metric_w1,
    "TailHillErr":   metric_tail,
    "VaR5Err":       metric_var,
    "ES5Err":        metric_es,
    "MMD":           metric_mmd,
    "AbsACFMaha":    metric_abs_acf,
    "SignACFMaha":   metric_sgn_acf,
    "SigW_L2_d2":    lambda x: metric_sigw_L2(x, 2),
    "SigW_L2_d3":    lambda x: metric_sigw_L2(x, 3),
    "SigW_L2_d4":    lambda x: metric_sigw_L2(x, 4),
    "SigW_Maha_d2":  lambda x: metric_sigw_Maha(x, 2),
    "SigW_Maha_d3":  lambda x: metric_sigw_Maha(x, 3),
    "SigW_Maha_d4":  lambda x: metric_sigw_Maha(x, 4),
}

# ---------------------------------------------------------------- score loop
print("Scoring real test windows ...")
N = len(test_windows)
scores_real = {m: np.empty(N) for m in METRICS}
t0 = time.time()
for i, w in enumerate(test_windows):
    for m, fn in METRICS.items():
        scores_real[m][i] = fn(w)
    if i and i % 500 == 0:
        print(f"  {i}/{N}  {time.time()-t0:.1f}s")
print(f"  done {time.time()-t0:.1f}s")

print("Scoring corrupted test windows ...")
scores_corrupt = {c: {m: np.empty(N) for m in METRICS} for c in CORRUPTIONS}
for c_name, c_fn in CORRUPTIONS.items():
    t1 = time.time()
    for i, w in enumerate(test_windows):
        wc = c_fn(w, np.random.default_rng(10_000 + i))
        for m, fn in METRICS.items():
            scores_corrupt[c_name][m][i] = fn(wc)
    print(f"  {c_name}: {time.time()-t1:.1f}s")

# ---------------------------------------------------------------- AUC analysis
metric_groups = {
    "marginal":  ["KS", "Wasserstein1", "TailHillErr", "VaR5Err", "ES5Err", "MMD"],
    "temporal":  ["AbsACFMaha", "SignACFMaha"],
    "signatureL2":   ["SigW_L2_d2", "SigW_L2_d3", "SigW_L2_d4"],
    "signatureMaha": ["SigW_Maha_d2", "SigW_Maha_d3", "SigW_Maha_d4"],
}
all_metrics = sum(metric_groups.values(), [])

auc_table = {m: {} for m in all_metrics}
for m in all_metrics:
    for c in CORRUPTIONS:
        sr, sc = scores_real[m], scores_corrupt[c][m]
        y = np.concatenate([np.zeros(len(sr)), np.ones(len(sc))])
        s = np.concatenate([sr, sc])
        auc_table[m][c] = float(roc_auc_score(y, s))

# Paired AUC
paired_auc_table = {m: {} for m in all_metrics}
for m in all_metrics:
    for c in CORRUPTIONS:
        sr = scores_real[m]; sc = scores_corrupt[c][m]
        gt = (sc > sr).sum()
        eq = (sc == sr).sum()
        paired_auc_table[m][c] = float((gt + 0.5*eq) / len(sr))

mean_auc = {m: float(np.mean(list(auc_table[m].values()))) for m in all_metrics}
paired_mean_auc = {m: float(np.mean(list(paired_auc_table[m].values()))) for m in all_metrics}

print("\nMean AUC per metric (unpaired):")
for m, v in sorted(mean_auc.items(), key=lambda x: -x[1]):
    print(f"  {m:18s} {v:.4f}")

print("\nMean paired AUC per metric:")
for m, v in sorted(paired_mean_auc.items(), key=lambda x: -x[1]):
    print(f"  {m:18s} {v:.4f}")

# Strongest marginal-only metric
best_marginal = max(metric_groups["marginal"], key=lambda m: mean_auc[m])
print("\nBest marginal-only metric:", best_marginal, "mean AUC =", mean_auc[best_marginal])

# Bootstrap: relative improvement of each signature variant over best_marginal
print("\nBootstrapping (paired AUC framing) ...")
B = 1000
def mean_paired_auc_idx(m, idx):
    aucs = []
    for c in CORRUPTIONS:
        sr = scores_real[m][idx]; sc = scores_corrupt[c][m][idx]
        gt = (sc > sr).sum(); eq = (sc == sr).sum()
        aucs.append((gt + 0.5*eq)/len(idx))
    return float(np.mean(aucs))

bootstrap = {}
N = len(test_windows)
for cand in metric_groups["signatureL2"] + metric_groups["signatureMaha"] + metric_groups["temporal"]:
    rels = []
    paired_aucs_sig = []
    paired_aucs_marg = []
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        m_sig  = mean_paired_auc_idx(cand, idx)
        m_marg = mean_paired_auc_idx(best_marginal, idx)
        paired_aucs_sig.append(m_sig)
        paired_aucs_marg.append(m_marg)
        if m_marg > 0:
            rels.append(100.0 * (m_sig - m_marg) / m_marg)
        else:
            rels.append(0.0)
    arr = np.array(rels)
    bootstrap[cand] = {
        "mean_paired_auc": float(np.mean(paired_aucs_sig)),
        "marginal_mean_paired_auc": float(np.mean(paired_aucs_marg)),
        "rel_improvement_pct_mean": float(arr.mean()),
        "rel_improvement_pct_lo": float(np.percentile(arr, 2.5)),
        "rel_improvement_pct_hi": float(np.percentile(arr, 97.5)),
        "ci_excludes_zero": bool(np.percentile(arr, 2.5) > 0 or np.percentile(arr, 97.5) < 0),
    }

print(json.dumps(bootstrap, indent=2))

# Verdict per acceptance criterion
print("\nAcceptance criterion: SUPPORTED if any Sig-W metric has mean relative")
print("improvement >= 3% over best marginal-only, 95% CI excluding zero.")
sig_candidates = metric_groups["signatureL2"] + metric_groups["signatureMaha"]
best_sig_name = max(sig_candidates, key=lambda m: bootstrap[m]["rel_improvement_pct_mean"])
best_sig_stats = bootstrap[best_sig_name]
print(f"  best signature variant: {best_sig_name}")
print(f"  mean rel improvement: {best_sig_stats['rel_improvement_pct_mean']:.2f}%")
print(f"  95% CI: [{best_sig_stats['rel_improvement_pct_lo']:.2f}, {best_sig_stats['rel_improvement_pct_hi']:.2f}]")

# Severity spearman
severity = {"time_reverse": 1, "block_shuffle": 2, "full_shuffle": 3}
sev_corr = {}
for m in all_metrics:
    pairs = []
    for c, lev in severity.items():
        gap = np.mean(scores_corrupt[c][m]) - np.mean(scores_real[m])
        pairs.append((lev, gap))
    if np.std([p[1] for p in pairs]) > 0:
        rho, _ = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
        sev_corr[m] = float(rho)
    else:
        sev_corr[m] = 0.0

# FNR at 5% FPR
fnr = {m: {} for m in all_metrics}
for m in all_metrics:
    for c in CORRUPTIONS:
        thr = np.quantile(scores_real[m], 0.95)
        fnr[m][c] = float(np.mean(scores_corrupt[c][m] <= thr))

# Runtime
print("\nRuntime benchmark ...")
t1 = time.time()
for w in test_windows[:200]:
    for d in SIG_DEPTHS:
        metric_sigw_Maha(w, d)
sig_rt = (time.time() - t1) / 200.0 * 1000.0
t1 = time.time()
for w in test_windows[:200]:
    metric_ks(w); metric_w1(w); metric_tail(w); metric_var(w); metric_es(w); metric_mmd(w)
marg_rt = (time.time() - t1) / 200.0 * 1000.0

out = {
    "auc_unpaired": auc_table,
    "auc_paired": paired_auc_table,
    "mean_auc_unpaired": mean_auc,
    "mean_auc_paired": paired_mean_auc,
    "best_marginal": {"name": best_marginal,
                      "mean_auc_unpaired": mean_auc[best_marginal],
                      "mean_auc_paired": paired_mean_auc[best_marginal]},
    "bootstrap": bootstrap,
    "best_signature": {"name": best_sig_name, **best_sig_stats},
    "severity_spearman": sev_corr,
    "fnr_at_5pct_fpr": fnr,
    "runtime_ms_per_1000_windows": {
        "sig_d2_d3_d4_maha_combined": float(sig_rt),
        "all_marginal_combined": float(marg_rt),
    },
    "n_ref_windows": int(len(ref_windows)),
    "n_test_windows": int(len(test_windows)),
    "n_tickers": int(close.shape[1]),
    "window_size": int(W),
    "stride": int(STRIDE),
    "corruptions": list(CORRUPTIONS),
    "sig_depths": SIG_DEPTHS,
}
with open("h3_results_v2.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nWrote h3_results_v2.json")
