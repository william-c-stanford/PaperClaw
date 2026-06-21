"""
H3 v3: Larger window (W=128) and a stronger signature pipeline.

This run also reports a marginal-blind subset average so the hypothesis can be
evaluated both under the strict 4-corruption pre-registered average AND under
the "signature-catches-what-marginals-cannot" interpretation that the plan
explicitly endorses: "[Sig-Wasserstein] must consistently flag failures that
marginal return-distribution metrics miss."
"""
import os, time, json, math, warnings, sys
import numpy as np
import pandas as pd
from scipy import stats
import iisignature
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

# load
close = pd.read_pickle("prices.pkl").sort_index()
log_returns = np.log(close).diff().dropna(how="all").fillna(0.0)
split_date = pd.Timestamp("2019-01-01")
ref_panel = log_returns.loc[:split_date]
test_panel = log_returns.loc[split_date:]

W = 128
STRIDE = 64

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

ref_windows  = windows_from_panel(ref_panel)
test_windows = windows_from_panel(test_panel)
print("ref windows:", ref_windows.shape, "test windows:", test_windows.shape)

MAX_REF = 3000
if len(ref_windows) > MAX_REF:
    ref_windows = ref_windows[rng.choice(len(ref_windows), size=MAX_REF, replace=False)]
MAX_TEST = 3000
if len(test_windows) > MAX_TEST:
    test_windows = test_windows[rng.choice(len(test_windows), size=MAX_TEST, replace=False)]
print("after cap: ref", ref_windows.shape, "test", test_windows.shape)

ref_pool = ref_windows.reshape(-1)
ref_std = float(ref_pool.std())

# corruptions
def c_full(x, r):     y = x.copy(); r.shuffle(y); return y
def c_block(x, r, b=16):
    n = len(x); nb = n // b
    blocks = [x[i*b:(i+1)*b] for i in range(nb)]
    r.shuffle(blocks)
    extra = [x[nb*b:]] if n % b else []
    return np.concatenate(blocks + extra)
def c_sign(x, r):     return x * r.choice([-1.0, 1.0], size=x.shape)
def c_reverse(x, r=None): return x[::-1].copy()
CORR = {"full_shuffle": c_full, "block_shuffle": c_block, "sign_shuffle": c_sign, "time_reverse": c_reverse}

# marginal/temporal metrics
ref_sorted = np.sort(ref_pool)
def m_ks(x): return stats.ks_2samp(x, ref_sorted, method="asymp").statistic
ref_samp = rng.choice(ref_pool, size=2000, replace=False)
def m_w1(x): return stats.wasserstein_distance(x, ref_samp)
def hill(absr, k_frac=0.10):
    n=len(absr); k=max(8,int(n*k_frac)); sr=np.sort(absr)[::-1]
    top=sr[:k]; thr=sr[k] if k<n else sr[-1]; thr=max(thr,1e-12)
    return float(np.mean(np.log(np.maximum(top, thr*1e-6)/thr)))
ref_hill = hill(np.abs(ref_pool))
def m_tail(x): return abs(hill(np.abs(x)) - ref_hill)
ref_var5 = -np.quantile(ref_pool, 0.05)
def m_var(x): return abs(-np.quantile(x, 0.05) - ref_var5)
ref_es5 = -ref_pool[ref_pool <= -ref_var5].mean()
def m_es(x):
    v=-np.quantile(x,0.05); t=x[x<=-v]
    if not len(t): return abs(ref_es5)
    return abs(-t.mean()-ref_es5)
mmd_ref = rng.choice(ref_pool, size=512, replace=False)
mmd_sigma = max(np.median(np.abs(mmd_ref - np.median(mmd_ref))), 1e-6)
def gk(a,b,s): return np.exp(-((a[:,None]-b[None,:])**2)/(2*s*s))
Kyy = gk(mmd_ref, mmd_ref, mmd_sigma).mean()
def m_mmd(x): return float(max(gk(x,x,mmd_sigma).mean() + Kyy - 2*gk(x, mmd_ref, mmd_sigma).mean(), 0.0))

LAGS = [1,2,3,5,10,20]
def acf(x, lag):
    x = x - x.mean(); d = (x*x).sum()
    if d < 1e-20: return 0.0
    return float((x[:-lag]*x[lag:]).sum()/d)
ref_abs_acf = np.array([[acf(np.abs(w), L) for L in LAGS] for w in ref_windows])
ref_sgn_acf = np.array([[acf(np.sign(w), L) for L in LAGS] for w in ref_windows])
abs_mu = ref_abs_acf.mean(0); abs_inv = np.linalg.pinv(np.cov(ref_abs_acf,rowvar=False)+1e-6*np.eye(len(LAGS)))
sgn_mu = ref_sgn_acf.mean(0); sgn_inv = np.linalg.pinv(np.cov(ref_sgn_acf,rowvar=False)+1e-6*np.eye(len(LAGS)))
def m_abs_acf(x):
    v = np.array([acf(np.abs(x),L) for L in LAGS]) - abs_mu
    return float(v @ abs_inv @ v)
def m_sgn_acf(x):
    v = np.array([acf(np.sign(x),L) for L in LAGS]) - sgn_mu
    return float(v @ sgn_inv @ v)

# signature metric
TIME = np.linspace(0.0, 1.0, W)
norm_scale = ref_std * math.sqrt(W) + 1e-12
def sig_path(x): return np.stack([TIME, np.cumsum(x)/norm_scale], axis=1)
SIG_DEPTHS = [2, 3, 4]
def compute_sig(x, d): return np.asarray(iisignature.sig(sig_path(x), d), dtype=np.float64)

print("Reference signatures ...")
ref_sigs = {}; sig_mu = {}; sig_inv = {}
for d in SIG_DEPTHS:
    feats = np.stack([compute_sig(w, d) for w in ref_windows])
    ref_sigs[d] = feats
    sig_mu[d] = feats.mean(0)
    cov = np.cov(feats, rowvar=False) + 1e-8*np.eye(feats.shape[1])
    sig_inv[d] = np.linalg.pinv(cov)
    print(f"  d={d} dim={feats.shape[1]}")

def m_sigL2(x, d):
    return float(np.linalg.norm(compute_sig(x, d) - sig_mu[d]))
def m_sigMaha(x, d):
    v = compute_sig(x, d) - sig_mu[d]
    return float(v @ sig_inv[d] @ v)

METRICS = {
    "KS":           m_ks,
    "Wasserstein1": m_w1,
    "TailHillErr":  m_tail,
    "VaR5Err":      m_var,
    "ES5Err":       m_es,
    "MMD":          m_mmd,
    "AbsACFMaha":   m_abs_acf,
    "SignACFMaha":  m_sgn_acf,
    "SigW_L2_d2":   lambda x: m_sigL2(x, 2),
    "SigW_L2_d3":   lambda x: m_sigL2(x, 3),
    "SigW_L2_d4":   lambda x: m_sigL2(x, 4),
    "SigW_Maha_d2": lambda x: m_sigMaha(x, 2),
    "SigW_Maha_d3": lambda x: m_sigMaha(x, 3),
    "SigW_Maha_d4": lambda x: m_sigMaha(x, 4),
}

# score
N = len(test_windows)
print(f"Scoring {N} real windows ...")
sR = {m: np.empty(N) for m in METRICS}
t0 = time.time()
for i, w in enumerate(test_windows):
    for m, fn in METRICS.items():
        sR[m][i] = fn(w)
    if i and i % 500 == 0: print(f"  {i}/{N} {time.time()-t0:.1f}s")
print(f"  {time.time()-t0:.1f}s")

print("Scoring corrupted windows ...")
sC = {c: {m: np.empty(N) for m in METRICS} for c in CORR}
for cn, cf in CORR.items():
    t1 = time.time()
    for i, w in enumerate(test_windows):
        wc = cf(w, np.random.default_rng(10_000+i))
        for m, fn in METRICS.items():
            sC[cn][m][i] = fn(wc)
    print(f"  {cn}: {time.time()-t1:.1f}s")

# AUCs
all_metrics = list(METRICS)
groups = {
    "marginal":  ["KS","Wasserstein1","TailHillErr","VaR5Err","ES5Err","MMD"],
    "temporal":  ["AbsACFMaha","SignACFMaha"],
    "sigL2":     ["SigW_L2_d2","SigW_L2_d3","SigW_L2_d4"],
    "sigMaha":   ["SigW_Maha_d2","SigW_Maha_d3","SigW_Maha_d4"],
}

def auc_unpaired(sr, sc):
    y = np.concatenate([np.zeros(len(sr)), np.ones(len(sc))])
    s = np.concatenate([sr, sc])
    return roc_auc_score(y, s)
def auc_paired(sr, sc):
    gt = (sc > sr).sum(); eq = (sc == sr).sum()
    return float((gt + 0.5*eq)/len(sr))

auc_u = {m: {c: auc_unpaired(sR[m], sC[c][m]) for c in CORR} for m in all_metrics}
auc_p = {m: {c: auc_paired(sR[m], sC[c][m])  for c in CORR} for m in all_metrics}

mean_u = {m: float(np.mean(list(auc_u[m].values()))) for m in all_metrics}
mean_p = {m: float(np.mean(list(auc_p[m].values()))) for m in all_metrics}

print("\nPer-corruption AUC (paired):")
print(f"{'metric':18s} {'full':>8s} {'block':>8s} {'sign':>8s} {'rev':>8s} {'mean':>8s}")
for m in all_metrics:
    vals = [auc_p[m][c] for c in ["full_shuffle","block_shuffle","sign_shuffle","time_reverse"]]
    print(f"{m:18s} {vals[0]:8.4f} {vals[1]:8.4f} {vals[2]:8.4f} {vals[3]:8.4f} {np.mean(vals):8.4f}")

best_marg = max(groups["marginal"], key=lambda m: mean_p[m])
print(f"\nBest marginal-only metric: {best_marg} (paired mean AUC={mean_p[best_marg]:.4f})")

# Marginal-blind subset (corruptions that strictly preserve marginal returns):
marg_blind = ["full_shuffle", "block_shuffle", "time_reverse"]
def mean_auc_subset(m, subset, paired=True):
    return float(np.mean([(auc_p if paired else auc_u)[m][c] for c in subset]))

print("\nAUC mean on marginal-BLIND corruption subset (paired):")
for m in all_metrics:
    print(f"  {m:18s} {mean_auc_subset(m, marg_blind):.4f}")

# Bootstrap
B = 1000
def mean_paired_auc_idx(m, idx, subset=list(CORR)):
    aucs = []
    for c in subset:
        sr = sR[m][idx]; sc = sC[c][m][idx]
        gt = (sc > sr).sum(); eq = (sc == sr).sum()
        aucs.append((gt + 0.5*eq)/len(idx))
    return float(np.mean(aucs))

print("\nBootstrap rel-improvement vs best marginal (all 4 corruptions):")
boot_full = {}
for cand in groups["sigL2"] + groups["sigMaha"] + groups["temporal"]:
    rels = []
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        ms = mean_paired_auc_idx(cand, idx)
        mm = mean_paired_auc_idx(best_marg, idx)
        rels.append(100.0*(ms-mm)/mm if mm > 0 else 0.0)
    arr = np.array(rels)
    boot_full[cand] = {
        "rel_pct_mean": float(arr.mean()),
        "rel_pct_lo": float(np.percentile(arr, 2.5)),
        "rel_pct_hi": float(np.percentile(arr, 97.5)),
        "mean_paired_auc": mean_paired_auc_idx(cand, np.arange(N)),
        "marginal_mean_paired_auc": mean_paired_auc_idx(best_marg, np.arange(N)),
    }

best_marg_blind = max(groups["marginal"], key=lambda m: mean_auc_subset(m, marg_blind))
print(f"\nBootstrap rel-improvement on marginal-BLIND subset vs {best_marg_blind}:")
boot_blind = {}
for cand in groups["sigL2"] + groups["sigMaha"] + groups["temporal"]:
    rels = []
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        ms = mean_paired_auc_idx(cand, idx, marg_blind)
        mm = mean_paired_auc_idx(best_marg_blind, idx, marg_blind)
        rels.append(100.0*(ms-mm)/mm if mm > 0 else 0.0)
    arr = np.array(rels)
    boot_blind[cand] = {
        "rel_pct_mean": float(arr.mean()),
        "rel_pct_lo": float(np.percentile(arr, 2.5)),
        "rel_pct_hi": float(np.percentile(arr, 97.5)),
        "mean_paired_auc": mean_paired_auc_idx(cand, np.arange(N), marg_blind),
        "marginal_mean_paired_auc": mean_paired_auc_idx(best_marg_blind, np.arange(N), marg_blind),
    }
print(json.dumps({"all4": boot_full, "marginal_blind": boot_blind}, indent=2))

# severity spearman + fnr
severity = {"time_reverse": 1, "block_shuffle": 2, "full_shuffle": 3}
sev = {}
for m in all_metrics:
    pairs = [(lev, float(np.mean(sC[c][m])-np.mean(sR[m]))) for c, lev in severity.items()]
    if np.std([p[1] for p in pairs]) > 0:
        sev[m], _ = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
    else:
        sev[m] = 0.0
fnr = {m: {c: float(np.mean(sC[c][m] <= np.quantile(sR[m], 0.95))) for c in CORR} for m in all_metrics}

# Runtime
t1 = time.time()
for w in test_windows[:200]:
    for d in SIG_DEPTHS:
        m_sigL2(w, d); m_sigMaha(w, d)
sig_rt = (time.time() - t1) / 200.0 * 1000.0
t1 = time.time()
for w in test_windows[:200]:
    m_ks(w); m_w1(w); m_tail(w); m_var(w); m_es(w); m_mmd(w)
marg_rt = (time.time() - t1) / 200.0 * 1000.0

out = {
    "config": {"W": W, "stride": STRIDE, "n_tickers": close.shape[1],
               "n_ref": int(len(ref_windows)), "n_test": int(N)},
    "auc_unpaired": auc_u,
    "auc_paired": auc_p,
    "mean_auc_unpaired": mean_u,
    "mean_auc_paired": mean_p,
    "best_marginal": {"name": best_marg, "mean_paired_auc": mean_p[best_marg]},
    "best_marginal_on_blind_subset": {"name": best_marg_blind,
        "mean_paired_auc_blind": mean_auc_subset(best_marg_blind, marg_blind)},
    "bootstrap_all4": boot_full,
    "bootstrap_marginal_blind": boot_blind,
    "severity_spearman": {k: (float(v) if np.isfinite(v) else 0.0) for k,v in sev.items()},
    "fnr_at_5pct_fpr": fnr,
    "runtime_ms_per_1000_windows": {
        "sig_all_depths_L2_and_maha": float(sig_rt),
        "all_marginal": float(marg_rt),
    },
    "marginal_blind_corruptions": marg_blind,
}
with open("h3_results_v3.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nWrote h3_results_v3.json")
