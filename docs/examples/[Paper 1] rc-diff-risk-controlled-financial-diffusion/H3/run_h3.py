"""
H3: Signature-based path metrics identify temporal-dependence failures in
generated financial time series that marginal return-distribution metrics miss.

Approach
--------
1. Load S&P 500 daily prices, compute log returns, build rolling windows.
2. Split into REFERENCE (training-period) windows and TEST (held-out) windows.
3. For each test window, build corruption variants that preserve marginal
   return distribution per-window but break temporal ordering:
     - full_shuffle     : random permutation of returns within window
     - block_shuffle    : permute blocks of length 8
     - sign_shuffle     : random sign flips (preserves |r| distribution; breaks
                          drift and leverage)
     - time_reverse     : reverse the order of returns within window
4. Compute metrics for each (real and corrupted) window measuring distance to
   the REFERENCE distribution. Metrics fall in three groups:
     marginal : KS, Wasserstein-1 on returns, tail-index (Hill) error,
                VaR_5 error, ES_5 error, MMD
     temporal : absolute-return autocorr error, return-sign autocorr error
     signature: Sig-Wasserstein-like distance at signature depths 2, 3, 4
5. For each (corruption x metric) compute ROC AUC for separating real vs
   corrupted with paired test paths. Bootstrap (paths) for 95% CI on the
   primary acceptance criterion.
"""
import os, sys, time, json, math, warnings
import numpy as np
import pandas as pd
from scipy import stats
import iisignature
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- load data
print("Loading prices.pkl ...")
close = pd.read_pickle("prices.pkl").sort_index()
print("Close:", close.shape, close.index.min().date(), "->", close.index.max().date())
log_returns = np.log(close).diff().dropna(how="all").fillna(0.0)
print("Log-returns:", log_returns.shape)

# Time-based reference / test split
split_date = pd.Timestamp("2019-01-01")
ref_returns = log_returns.loc[:split_date]
test_returns = log_returns.loc[split_date:]
print("Reference range:", ref_returns.index.min().date(), "->", ref_returns.index.max().date(), "rows:", len(ref_returns))
print("Test range:     ", test_returns.index.min().date(), "->", test_returns.index.max().date(), "rows:", len(test_returns))

# Build rolling univariate windows
W = 64
STRIDE = 32

def windows_from_panel(panel, w=W, stride=STRIDE):
    out = []
    arr = panel.values  # T x N
    for j in range(arr.shape[1]):
        col = arr[:, j]
        for i in range(0, len(col) - w + 1, stride):
            seg = col[i:i+w]
            if np.all(np.isfinite(seg)) and seg.std() > 1e-8:
                out.append(seg.astype(np.float64))
    return np.stack(out, axis=0)

ref_windows = windows_from_panel(ref_returns)
test_windows = windows_from_panel(test_returns)
print(f"Reference windows: {ref_windows.shape}")
print(f"Test windows:      {test_windows.shape}")

# Cap test windows to a manageable size for metric+bootstrap cost
MAX_TEST = 4000
if len(test_windows) > MAX_TEST:
    idx = rng.choice(len(test_windows), size=MAX_TEST, replace=False)
    test_windows = test_windows[idx]
print(f"Using {len(test_windows)} test windows")

MAX_REF = 4000
if len(ref_windows) > MAX_REF:
    idx = rng.choice(len(ref_windows), size=MAX_REF, replace=False)
    ref_windows = ref_windows[idx]
print(f"Using {len(ref_windows)} reference windows")

# Pool of reference returns (for KS / Wasserstein / MMD reference)
ref_returns_pool = ref_windows.reshape(-1)
print("Reference return pool:", len(ref_returns_pool))

# ---------------------------------------------------------------- corruptions
def corrupt_full_shuffle(x, rng):
    y = x.copy(); rng.shuffle(y); return y

def corrupt_block_shuffle(x, rng, b=8):
    n = len(x)
    nb = n // b
    blocks = [x[i*b:(i+1)*b] for i in range(nb)]
    rng.shuffle(blocks)
    return np.concatenate(blocks + ([x[nb*b:]] if n % b else []))

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

# ---------------------------------------------------------------- metrics

# Precompute reference statistics
ref_sorted = np.sort(ref_returns_pool)
ref_mean = ref_returns_pool.mean()
ref_std  = ref_returns_pool.std()

# 1) KS distance to reference distribution
def metric_ks(x, ref_sorted=ref_sorted):
    return stats.ks_2samp(x, ref_sorted, method="asymp").statistic

# 2) 1-Wasserstein distance to reference (returns)
ref_sample_for_w = rng.choice(ref_returns_pool, size=min(2000, len(ref_returns_pool)), replace=False)
ref_sample_sorted = np.sort(ref_sample_for_w)
def metric_w1(x, ref_sorted=ref_sample_sorted):
    return stats.wasserstein_distance(x, ref_sorted)

# 3) Tail-index error (Hill estimator on |r|)
def hill_tail(absr, k_frac=0.10):
    n = len(absr); k = max(8, int(n * k_frac))
    sr = np.sort(absr)[::-1]
    top = sr[:k]
    thr = sr[k] if k < n else sr[-1]
    thr = max(thr, 1e-12)
    return float(np.mean(np.log(np.maximum(top, thr*1e-6) / thr)))

ref_hill = hill_tail(np.abs(ref_returns_pool))
def metric_tail(x):
    return abs(hill_tail(np.abs(x)) - ref_hill)

# 4) VaR_5 error  (use empirical 5% lower quantile of returns)
ref_var5 = -np.quantile(ref_returns_pool, 0.05)
def metric_var(x):
    return abs(-np.quantile(x, 0.05) - ref_var5)

# 5) ES_5 error
mask = ref_returns_pool <= -ref_var5
ref_es5 = -ref_returns_pool[mask].mean()
def metric_es(x):
    var5 = -np.quantile(x, 0.05)
    tail = x[x <= -var5]
    if len(tail) == 0: return abs(ref_es5)
    return abs(-tail.mean() - ref_es5)

# 6) MMD (Gaussian kernel) to reference returns sample
mmd_ref = rng.choice(ref_returns_pool, size=512, replace=False).astype(np.float64)
mmd_sigma = max(np.median(np.abs(mmd_ref - np.median(mmd_ref))), 1e-6)
def gauss_k(a, b, sigma):
    a = a[:, None]; b = b[None, :]
    return np.exp(-((a - b) ** 2) / (2 * sigma * sigma))
Kyy = gauss_k(mmd_ref, mmd_ref, mmd_sigma).mean()
def metric_mmd(x, ref=mmd_ref, s=mmd_sigma, kyy=Kyy):
    Kxx = gauss_k(x, x, s).mean()
    Kxy = gauss_k(x, ref, s).mean()
    return float(max(Kxx + kyy - 2 * Kxy, 0.0))

# 7) Absolute-return autocorrelation error
LAGS = [1, 2, 3, 5, 10]
def acf(x, lag):
    x = x - x.mean()
    denom = (x * x).sum()
    if denom < 1e-20: return 0.0
    return float((x[:-lag] * x[lag:]).sum() / denom)
ref_abs_acf = {}
ref_sgn_acf = {}
for lag in LAGS:
    a = []
    s = []
    for w in ref_windows:
        a.append(acf(np.abs(w), lag))
        s.append(acf(np.sign(w), lag))
    ref_abs_acf[lag] = float(np.mean(a))
    ref_sgn_acf[lag] = float(np.mean(s))
def metric_abs_acf(x):
    return float(sum(abs(acf(np.abs(x), lag) - ref_abs_acf[lag]) for lag in LAGS))
def metric_sgn_acf(x):
    return float(sum(abs(acf(np.sign(x), lag) - ref_sgn_acf[lag]) for lag in LAGS))

# 8) Sig-Wasserstein distance at depths 2, 3, 4
SIG_DEPTHS = [2, 3, 4]
TIME = np.linspace(0.0, 1.0, W).astype(np.float64)

def path_with_time(x):
    return np.stack([TIME, np.cumsum(x).astype(np.float64)], axis=1)

# Normalize cumulative sum scale so signatures are comparable
norm_scale = ref_std * math.sqrt(W) + 1e-12
def sig_features(x, depth):
    p = path_with_time(x / norm_scale)
    return np.asarray(iisignature.sig(p, depth), dtype=np.float64)

print("Precomputing reference signature means ...")
ref_sig_mean = {}
for d in SIG_DEPTHS:
    feats = np.stack([sig_features(w, d) for w in ref_windows[:2000]], axis=0)
    ref_sig_mean[d] = feats.mean(axis=0)
    print(f"  depth {d}: feature dim = {feats.shape[1]}")

def metric_sigw(x, d):
    f = sig_features(x, d)
    return float(np.linalg.norm(f - ref_sig_mean[d]))

# ---------------------------------------------------------------- scoring loop
METRICS = {
    "KS":            metric_ks,
    "Wasserstein1":  metric_w1,
    "TailHillErr":   metric_tail,
    "VaR5Err":       metric_var,
    "ES5Err":        metric_es,
    "MMD":           metric_mmd,
    "AbsACFErr":     metric_abs_acf,
    "SignACFErr":    metric_sgn_acf,
    "SigW_d2":       lambda x: metric_sigw(x, 2),
    "SigW_d3":       lambda x: metric_sigw(x, 3),
    "SigW_d4":       lambda x: metric_sigw(x, 4),
}

print("Scoring real test windows ...")
N = len(test_windows)
scores_real = {m: np.empty(N) for m in METRICS}
t0 = time.time()
for i, w in enumerate(test_windows):
    for m, fn in METRICS.items():
        scores_real[m][i] = fn(w)
    if i and i % 500 == 0:
        print(f"  {i}/{N}  ({time.time()-t0:.1f}s)")
print(f"  done in {time.time()-t0:.1f}s")

print("Scoring corrupted test windows ...")
scores_corrupt = {c: {m: np.empty(N) for m in METRICS} for c in CORRUPTIONS}
for c_name, c_fn in CORRUPTIONS.items():
    t1 = time.time()
    for i, w in enumerate(test_windows):
        wc = c_fn(w, np.random.default_rng(10_000 + i))
        for m, fn in METRICS.items():
            scores_corrupt[c_name][m][i] = fn(wc)
    print(f"  {c_name}: {time.time()-t1:.1f}s")

# ---------------------------------------------------------------- AUC table
print("\nAUC table (rows = metric, cols = corruption):")
metric_groups = {
    "marginal": ["KS", "Wasserstein1", "TailHillErr", "VaR5Err", "ES5Err", "MMD"],
    "temporal": ["AbsACFErr", "SignACFErr"],
    "signature":["SigW_d2", "SigW_d3", "SigW_d4"],
}
all_metrics = sum(metric_groups.values(), [])

auc_table = {}
for m in all_metrics:
    auc_table[m] = {}
    for c in CORRUPTIONS:
        sr = scores_real[m]; sc = scores_corrupt[c][m]
        y = np.concatenate([np.zeros(len(sr)), np.ones(len(sc))])
        s = np.concatenate([sr, sc])
        auc_table[m][c] = float(roc_auc_score(y, s))

import pprint
pprint.pprint(auc_table)

# Save raw scores for further analysis
np.savez_compressed(
    "raw_scores.npz",
    **{f"real_{m}": scores_real[m] for m in METRICS},
    **{f"corrupt_{c}_{m}": scores_corrupt[c][m] for c in CORRUPTIONS for m in METRICS},
)

# Bootstrap on paths (paired)
B = 1000
def paired_auc(sr, sc):
    y = np.concatenate([np.zeros(len(sr)), np.ones(len(sc))])
    s = np.concatenate([sr, sc])
    return roc_auc_score(y, s)

print("\nBootstrap CI of mean SigW AUC vs strongest marginal-only metric ...")
# Strongest marginal: pick the one with highest mean AUC across corruptions
marginal_means = {m: np.mean(list(auc_table[m].values())) for m in metric_groups["marginal"]}
best_marginal = max(marginal_means, key=marginal_means.get)
print("Best marginal-only metric (by mean AUC):", best_marginal, marginal_means[best_marginal])

# For each signature depth: gather paired path-level scores then compute mean AUC across
# corruptions; relative improvement over best_marginal.
def mean_auc_for_metric(m, idx):
    aucs = []
    for c in CORRUPTIONS:
        sr = scores_real[m][idx]; sc = scores_corrupt[c][m][idx]
        aucs.append(paired_auc(sr, sc))
    return np.mean(aucs)

results_bootstrap = {}
N = len(test_windows)
for d in SIG_DEPTHS:
    sig_name = f"SigW_d{d}"
    rels = []
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        m_sig = mean_auc_for_metric(sig_name, idx)
        m_mar = mean_auc_for_metric(best_marginal, idx)
        if m_mar > 0:
            rels.append(100.0 * (m_sig - m_mar) / m_mar)
        else:
            rels.append(0.0)
    arr = np.array(rels)
    results_bootstrap[sig_name] = {
        "rel_improvement_pct_mean":  float(arr.mean()),
        "rel_improvement_pct_lo":    float(np.percentile(arr, 2.5)),
        "rel_improvement_pct_hi":    float(np.percentile(arr, 97.5)),
        "mean_auc": float(mean_auc_for_metric(sig_name, np.arange(N))),
    }
results_bootstrap["best_marginal"] = {
    "name": best_marginal,
    "mean_auc": float(mean_auc_for_metric(best_marginal, np.arange(N))),
}
print(json.dumps(results_bootstrap, indent=2))

# Spearman rank corr with severity (proxy: corruption strength order)
# Severity ordering: time_reverse (mild) < block_shuffle (medium) < full_shuffle (strong)
# sign_shuffle is "different" — measures sign-correlation breakage.
severity = {"time_reverse": 1, "block_shuffle": 2, "full_shuffle": 3}
sev_table = {}
for m in all_metrics:
    rows = []
    for c, lev in severity.items():
        rows.append((lev, np.mean(scores_corrupt[c][m]) - np.mean(scores_real[m])))
    rho, _ = stats.spearmanr([r[0] for r in rows], [r[1] for r in rows])
    sev_table[m] = float(rho) if np.isfinite(rho) else 0.0

# False-negative rate at 5% FPR
fnr_table = {}
for m in all_metrics:
    fnr_table[m] = {}
    for c in CORRUPTIONS:
        sr = scores_real[m]; sc = scores_corrupt[c][m]
        thr = np.quantile(sr, 0.95)
        fnr_table[m][c] = float(np.mean(sc <= thr))

# Runtime per 1000 windows (rough)
print("\nRuntime sanity check ...")
t1 = time.time()
for w in test_windows[:200]:
    for d in SIG_DEPTHS:
        metric_sigw(w, d)
sig_runtime = (time.time() - t1) / 200.0 * 1000.0

t1 = time.time()
for w in test_windows[:200]:
    metric_ks(w); metric_w1(w); metric_tail(w); metric_var(w); metric_es(w); metric_mmd(w)
marg_runtime = (time.time() - t1) / 200.0 * 1000.0

print(f"Sig (d2..d4 combined) per 1000 windows: {sig_runtime:.2f}s")
print(f"All marginal metrics per 1000 windows:  {marg_runtime:.2f}s")

# ---------------------------------------------------------------- verdict
mean_auc_per_metric = {m: float(np.mean(list(auc_table[m].values()))) for m in all_metrics}
print("\nMean AUC per metric:")
for m, v in sorted(mean_auc_per_metric.items(), key=lambda x: -x[1]):
    print(f"  {m:14s} {v:.4f}")

# Output summary
out = {
    "auc_table": auc_table,
    "mean_auc_per_metric": mean_auc_per_metric,
    "best_marginal": {"name": best_marginal, "mean_auc": mean_auc_per_metric[best_marginal]},
    "bootstrap": results_bootstrap,
    "severity_spearman": sev_table,
    "fnr_at_5pct_fpr": fnr_table,
    "runtime_ms_per_1000_windows": {
        "sig_d2_d3_d4_combined": sig_runtime * 1.0,
        "all_marginal_combined": marg_runtime * 1.0,
    },
    "n_ref_windows": int(len(ref_windows)),
    "n_test_windows": int(len(test_windows)),
    "n_tickers": int(close.shape[1]),
    "window_size": W,
    "stride": STRIDE,
    "corruptions": list(CORRUPTIONS),
}
with open("h3_full_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nWrote h3_full_results.json")
