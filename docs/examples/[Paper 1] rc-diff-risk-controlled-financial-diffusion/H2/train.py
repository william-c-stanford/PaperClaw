"""Train diffusion variants + GAN baselines, then evaluate volatility monotonicity (H2 primary)."""
import os, sys, json, time, pickle, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau

from models import (
    CondDiffusionUNet, DiffusionScheduler,
    GRUGen, GRUDisc, LSTMGen, LSTMDisc,
)

OUT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRADING_DAYS = 252

# ---------------- data ----------------

def load_data():
    with open(os.path.join(OUT, "data.pkl"), "rb") as f:
        d = pickle.load(f)
    return d


def stack_windows(per_asset):
    """Return tensor of all (N, win) windows + per-window asset labels."""
    arrs, labels = [], []
    syms = sorted(per_asset.keys())
    for ai, s in enumerate(syms):
        W = per_asset[s]
        arrs.append(W)
        labels.append(np.full(W.shape[0], ai, dtype=np.int64))
    X = np.concatenate(arrs, axis=0)
    A = np.concatenate(labels, axis=0)
    return X, A, syms


def realized_vol(x, win_days):
    """Annualized realized volatility of a window of log returns."""
    sd = x.std(axis=-1)
    return sd * math.sqrt(TRADING_DAYS)


def normalize(X, mu=None, sd=None):
    if mu is None:
        mu = X.mean()
        sd = X.std() + 1e-8
    return (X - mu) / sd, mu, sd


# ---------------- stylized-fact loss ----------------

def absret_autocorr(x, lag=1):
    """Lag-k autocorrelation of |returns| (vol clustering proxy)."""
    a = x.abs()
    a = a - a.mean(dim=-1, keepdim=True)
    num = (a[:, :-lag] * a[:, lag:]).mean(dim=-1)
    den = (a * a).mean(dim=-1) + 1e-8
    return num / den


def tail_index_proxy(x):
    """Hill-style proxy: ratio of |returns| above 95th percentile to overall mean — heavy tails increase this."""
    abs_x = x.abs()
    q = abs_x.quantile(0.95, dim=-1, keepdim=True)
    mask = (abs_x >= q).float()
    extreme = (abs_x * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    return extreme / (abs_x.mean(dim=-1) + 1e-8)


def stylized_fact_loss(x_pred, x_real):
    """Match marginal moments + abs-autocorr + tail-index proxy between predicted and real returns."""
    losses = []
    losses.append(((x_pred.std(dim=-1) - x_real.std(dim=-1)) ** 2).mean())
    losses.append(((x_pred.abs().mean(dim=-1) - x_real.abs().mean(dim=-1)) ** 2).mean())
    losses.append(((absret_autocorr(x_pred, 1) - absret_autocorr(x_real, 1)) ** 2).mean())
    losses.append(((absret_autocorr(x_pred, 5) - absret_autocorr(x_real, 5)) ** 2).mean())
    losses.append(((tail_index_proxy(x_pred) - tail_index_proxy(x_real)) ** 2).mean() * 0.1)
    return sum(losses)


# ---------------- diffusion training ----------------

def train_diffusion(X, vol_target, *, variant, seed, steps=6000, batch=256,
                    use_cond=True, sf_weight=0.0, shuffled_vol=False,
                    log_every=600):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CondDiffusionUNet(win=X.shape[1], ch=64, use_cond=use_cond).to(DEVICE)
    sched = DiffusionScheduler(T=200, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    X_t = torch.from_numpy(X).to(DEVICE)
    V_t = torch.from_numpy(vol_target).to(DEVICE)
    if shuffled_vol:
        perm = torch.randperm(V_t.size(0))
        V_t = V_t[perm]
    n = X_t.size(0)
    losses = []
    model.train()
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=DEVICE)
        x0 = X_t[idx]
        cond = V_t[idx] if use_cond else None
        t = torch.randint(0, sched.T, (batch,), device=DEVICE)
        noise = torch.randn_like(x0)
        x_t = sched.q_sample(x0, t, noise)
        eps_pred = model(x_t, t, cond=cond)
        loss = F.mse_loss(eps_pred, noise)
        if sf_weight > 0:
            # approximate predicted x0 to compute stylized-fact loss
            sa = sched.sqrt_ab[t].unsqueeze(-1)
            so = sched.sqrt_one_minus_ab[t].unsqueeze(-1)
            x0_hat = (x_t - so * eps_pred) / (sa + 1e-8)
            loss = loss + sf_weight * stylized_fact_loss(x0_hat, x0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            losses.append((step, float(loss.item())))
    model.eval()
    return model, sched, losses


# ---------------- GAN baseline training ----------------

def train_timegan_lite(X, *, seed, steps=4000, batch=256, cond_dim=0, V=None):
    """Compact TimeGAN-style: GRU generator vs GRU discriminator + supervised next-step loss."""
    torch.manual_seed(seed); np.random.seed(seed)
    win = X.shape[1]
    G = GRUGen(latent=16, hidden=64, win=win, cond_dim=cond_dim).to(DEVICE)
    D = GRUDisc(hidden=64, win=win, cond_dim=cond_dim).to(DEVICE)
    # supervisor: predict next step
    Sup = nn.GRU(1, 64, batch_first=True, num_layers=2).to(DEVICE)
    Sup_head = nn.Linear(64, 1).to(DEVICE)
    optG = torch.optim.Adam(list(G.parameters()) + list(Sup.parameters()) + list(Sup_head.parameters()), lr=2e-4)
    optD = torch.optim.Adam(D.parameters(), lr=2e-4)
    X_t = torch.from_numpy(X).to(DEVICE)
    V_t = torch.from_numpy(V).to(DEVICE) if V is not None else None
    bce = nn.BCEWithLogitsLoss()
    n = X_t.size(0)
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=DEVICE)
        x_real = X_t[idx]
        c_real = V_t[idx] if (cond_dim and V_t is not None) else None
        # train D
        with torch.no_grad():
            x_fake = G(batch, DEVICE, cond=c_real)
        d_real = D(x_real, cond=c_real)
        d_fake = D(x_fake, cond=c_real)
        loss_D = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        optD.zero_grad(); loss_D.backward(); optD.step()
        # train G
        x_fake = G(batch, DEVICE, cond=c_real)
        d_fake = D(x_fake, cond=c_real)
        # supervised: train supervisor to predict next step from real
        h_real, _ = Sup(x_real.unsqueeze(-1))
        pred_real = Sup_head(h_real[:, :-1, :]).squeeze(-1)
        sup_loss = F.mse_loss(pred_real, x_real[:, 1:])
        # supervised regularization on fake
        h_fake, _ = Sup(x_fake.unsqueeze(-1))
        pred_fake = Sup_head(h_fake[:, :-1, :]).squeeze(-1)
        sup_fake = F.mse_loss(pred_fake, x_fake[:, 1:].detach())
        loss_G = bce(d_fake, torch.ones_like(d_fake)) + 10.0 * sup_loss + 5.0 * sup_fake
        optG.zero_grad(); loss_G.backward(); optG.step()
    G.eval(); D.eval()
    return G


def train_crnn_gan(X, *, seed, steps=4000, batch=256):
    torch.manual_seed(seed); np.random.seed(seed)
    win = X.shape[1]
    G = LSTMGen(latent=8, hidden=64, win=win).to(DEVICE)
    D = LSTMDisc(hidden=64, win=win).to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=2e-4)
    optD = torch.optim.Adam(D.parameters(), lr=2e-4)
    X_t = torch.from_numpy(X).to(DEVICE)
    bce = nn.BCEWithLogitsLoss()
    n = X_t.size(0)
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=DEVICE)
        x_real = X_t[idx]
        with torch.no_grad():
            x_fake = G(batch, DEVICE)
        d_real = D(x_real); d_fake = D(x_fake)
        loss_D = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        optD.zero_grad(); loss_D.backward(); optD.step()
        x_fake = G(batch, DEVICE)
        d_fake = D(x_fake)
        loss_G = bce(d_fake, torch.ones_like(d_fake))
        optG.zero_grad(); loss_G.backward(); optG.step()
    G.eval(); D.eval()
    return G


def train_sigw_gan(X, V, *, seed, steps=4000, batch=256):
    """Sig-W-GAN-lite: conditional GAN whose generator+discriminator loss includes a
    signature-style truncated path moment-matching penalty (cumulative sums to truncation level 3).
    This stands in for the path-signature objective of Liao et al. 2023."""
    torch.manual_seed(seed); np.random.seed(seed)
    win = X.shape[1]
    G = GRUGen(latent=16, hidden=64, win=win, cond_dim=1).to(DEVICE)
    D = GRUDisc(hidden=64, win=win, cond_dim=1).to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=2e-4)
    optD = torch.optim.Adam(D.parameters(), lr=2e-4)
    X_t = torch.from_numpy(X).to(DEVICE)
    V_t = torch.from_numpy(V).to(DEVICE)
    bce = nn.BCEWithLogitsLoss()
    n = X_t.size(0)

    def sig_loss(a, b):
        # Truncated path-signature proxy: moments of cumulative sums up to order 3.
        ca = torch.cumsum(a, dim=-1); cb = torch.cumsum(b, dim=-1)
        L = 0.0
        for k in [1, 2, 3]:
            L = L + ((ca ** k).mean(dim=-1).mean() - (cb ** k).mean(dim=-1).mean()) ** 2
        return L

    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=DEVICE)
        x_real = X_t[idx]; c_real = V_t[idx]
        with torch.no_grad():
            x_fake = G(batch, DEVICE, cond=c_real)
        d_real = D(x_real, cond=c_real); d_fake = D(x_fake, cond=c_real)
        loss_D = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        optD.zero_grad(); loss_D.backward(); optD.step()
        x_fake = G(batch, DEVICE, cond=c_real)
        d_fake = D(x_fake, cond=c_real)
        sw = sig_loss(x_real, x_fake)
        loss_G = bce(d_fake, torch.ones_like(d_fake)) + 5.0 * sw
        optG.zero_grad(); loss_G.backward(); optG.step()
    G.eval(); D.eval()
    return G


# ---------------- generation ----------------

@torch.no_grad()
def diffusion_generate(model, sched, n, win, cond=None, batch=512):
    out = []
    for i in range(0, n, batch):
        m = min(batch, n - i)
        c = cond[i:i+m] if cond is not None else None
        x = sched.sample(model, m, win, DEVICE, cond=c)
        out.append(x.cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def gan_generate(G, n, win, cond=None, batch=512):
    out = []
    for i in range(0, n, batch):
        m = min(batch, n - i)
        c = cond[i:i+m] if cond is not None else None
        x = G(m, DEVICE, cond=c) if hasattr(G, "cond_dim") and G.cond_dim > 0 and cond is not None else G(m, DEVICE)
        out.append(x.cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------- evaluation ----------------

def primary_spearman(model_kind, gen_fn, target_quantiles, n_per_q, win, mu, sd):
    """Sample conditioned generations across target volatility quantiles, return Spearman + per-bin stats.
    For unconditional models, gen_fn ignores cond and we use post-hoc binning of generated samples."""
    rng = np.random.default_rng(0)
    requested_vols, realized_vols, bin_ids = [], [], []
    if model_kind == "conditional":
        for qi, q in enumerate(target_quantiles):
            cond = torch.full((n_per_q,), float(q), device=DEVICE, dtype=torch.float32)
            x_gen = gen_fn(n_per_q, win, cond=cond)
            # denormalize
            x_gen = x_gen * sd + mu
            v = np.std(x_gen, axis=-1) * math.sqrt(TRADING_DAYS)
            requested_vols.extend([float(q)] * n_per_q)
            realized_vols.extend(v.tolist())
            bin_ids.extend([qi] * n_per_q)
    else:
        # unconditional + post-hoc binning: sample a pool, sort by realized vol, assign quantile bins
        pool_size = n_per_q * len(target_quantiles)
        x_gen = gen_fn(pool_size, win, cond=None)
        x_gen = x_gen * sd + mu
        v = np.std(x_gen, axis=-1) * math.sqrt(TRADING_DAYS)
        order = np.argsort(v)
        bins = np.array_split(order, len(target_quantiles))
        for qi, b in enumerate(bins):
            requested_vols.extend([float(target_quantiles[qi])] * len(b))
            realized_vols.extend(v[b].tolist())
            bin_ids.extend([qi] * len(b))
    requested_vols = np.array(requested_vols)
    realized_vols = np.array(realized_vols)
    bin_ids = np.array(bin_ids)
    spear, _ = spearmanr(requested_vols, realized_vols)
    # per-bin medians for monotonicity-violation rate
    bin_medians = np.array([np.median(realized_vols[bin_ids == i]) for i in range(len(target_quantiles))])
    monot_violations = float(np.mean(np.diff(bin_medians) < 0))
    return dict(
        spearman=float(spear),
        bin_medians=bin_medians.tolist(),
        monot_violation_rate=monot_violations,
        requested=requested_vols.tolist()[:200],  # small slice for inspection
        realized=realized_vols.tolist()[:200],
    ), requested_vols, realized_vols, bin_ids


def calibration_metrics(model_kind, gen_fn, target_levels, n_per_q, win, mu, sd, real_pool):
    """Return calibration errors for VaR (5%), ES, and max-drawdown vs requested vol levels.
    For each requested vol level (annualized), compute target absolute risk values using a Gaussian
    benchmark with that vol; measure absolute difference from generated paths' empirical risk."""
    out = {}
    var_err, es_err, mdd_err = [], [], []
    # Generate one big pool we can stratify
    if model_kind == "conditional":
        gens, reqs = [], []
        for q in target_levels:
            cond = torch.full((n_per_q,), float(q), device=DEVICE, dtype=torch.float32)
            x_gen = gen_fn(n_per_q, win, cond=cond)
            x_gen = x_gen * sd + mu
            gens.append(x_gen); reqs.extend([q] * n_per_q)
        gens = np.concatenate(gens, axis=0); reqs = np.array(reqs)
    else:
        pool_size = n_per_q * len(target_levels)
        x_gen = gen_fn(pool_size, win, cond=None)
        x_gen = x_gen * sd + mu
        v = np.std(x_gen, axis=-1) * math.sqrt(TRADING_DAYS)
        order = np.argsort(v)
        bins = np.array_split(order, len(target_levels))
        gens_chunks, reqs = [], []
        for qi, b in enumerate(bins):
            gens_chunks.append(x_gen[b])
            reqs.extend([target_levels[qi]] * len(b))
        gens = np.concatenate(gens_chunks, axis=0); reqs = np.array(reqs)

    for q in target_levels:
        mask = reqs == q
        x = gens[mask]  # (n, win)
        if x.size == 0:
            continue
        # generated empirical metrics
        var5 = -np.quantile(x, 0.05, axis=-1)
        n_tail = max(1, int(round(0.05 * x.shape[1])))
        sorted_x = np.sort(x, axis=-1)
        es5 = -sorted_x[:, :n_tail].mean(axis=-1)
        # max drawdown via cumulative
        cum = np.cumsum(x, axis=-1)
        run_max = np.maximum.accumulate(cum, axis=-1)
        dd = run_max - cum
        mdd = np.max(dd, axis=-1)
        # target: Gaussian benchmark with daily sd from annualized vol
        sd_daily = q / math.sqrt(TRADING_DAYS)
        target_var = 1.645 * sd_daily
        # Gaussian ES: phi(z) / (1-Phi(z)) * sd  at z=1.645 -> 2.063*sd
        target_es = 2.063 * sd_daily
        # crude Brownian-motion approx for mean abs max-drawdown of N steps: ~0.8 * sd * sqrt(N)
        target_mdd = 0.8 * sd_daily * math.sqrt(win)
        var_err.append(float(np.abs(var5.mean() - target_var) / max(target_var, 1e-8)))
        es_err.append(float(np.abs(es5.mean() - target_es) / max(target_es, 1e-8)))
        mdd_err.append(float(np.abs(mdd.mean() - target_mdd) / max(target_mdd, 1e-8)))
    out["var_calib_err"] = float(np.mean(var_err))
    out["es_calib_err"] = float(np.mean(es_err))
    out["mdd_calib_err"] = float(np.mean(mdd_err))
    # autocorr/abs-autocorr error & sig-W distance to real pool
    real_sub = real_pool[np.random.default_rng(0).choice(real_pool.shape[0], size=min(2000, real_pool.shape[0]), replace=False)]
    gen_sub = gens[np.random.default_rng(1).choice(gens.shape[0], size=min(2000, gens.shape[0]), replace=False)]
    def absacf(a, lag=1):
        a = np.abs(a)
        a = a - a.mean(axis=-1, keepdims=True)
        return ((a[:, :-lag] * a[:, lag:]).mean(-1) / (a.var(-1) + 1e-8)).mean()
    out["abs_autocorr_err_lag1"] = float(abs(absacf(gen_sub) - absacf(real_sub)))
    out["abs_autocorr_err_lag5"] = float(abs(absacf(gen_sub, 5) - absacf(real_sub, 5)))
    # signature-W proxy via cumulative-sum moments
    def sig_moments(a):
        c = np.cumsum(a, axis=-1)
        return np.array([c.mean(), (c ** 2).mean(), (c ** 3).mean(), (c ** 4).mean()])
    out["sigW_proxy"] = float(np.linalg.norm(sig_moments(gen_sub) - sig_moments(real_sub)))
    # tail index error (Hill-style)
    def tail(a):
        ab = np.abs(a); q95 = np.quantile(ab, 0.95, axis=-1, keepdims=True)
        m = ab >= q95
        ex = (ab * m).sum(-1) / (m.sum(-1) + 1e-8)
        return (ex / (ab.mean(-1) + 1e-8)).mean()
    out["tail_idx_err"] = float(abs(tail(gen_sub) - tail(real_sub)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--gan_steps", type=int, default=4000)
    parser.add_argument("--n_per_q", type=int, default=600)
    args = parser.parse_args()

    print(f"[device] {DEVICE}")
    data = load_data()
    per_asset = data["per_asset"]
    win = data["win"]
    rets_aligned = data["rets_aligned"]

    # Separate S&P 500 constituents windows from SPY
    sp500_keys = [k for k in per_asset.keys() if k != "SPY"]
    spy_W = per_asset.get("SPY")
    X_sp, A_sp, syms_sp = stack_windows({k: per_asset[k] for k in sp500_keys})
    X_spy = spy_W.copy()
    print(f"[data] sp500 windows={X_sp.shape}, spy windows={X_spy.shape}")

    # Compute conditioning labels (annualized vol of each window) on raw returns,
    # then normalize returns to a unit-variance pre-scaled tensor for training.
    vol_sp = (X_sp.std(axis=-1) * math.sqrt(TRADING_DAYS)).astype(np.float32)
    vol_spy = (X_spy.std(axis=-1) * math.sqrt(TRADING_DAYS)).astype(np.float32)

    # global standardization (so the diffusion model operates on near-unit-variance signals)
    X_sp_n, mu_sp, sd_sp = normalize(X_sp)
    X_spy_n, mu_spy, sd_spy = normalize(X_spy)
    X_sp_n = X_sp_n.astype(np.float32); X_spy_n = X_spy_n.astype(np.float32)
    print(f"[norm] sp mu={mu_sp:.3e} sd={sd_sp:.3e}  spy mu={mu_spy:.3e} sd={sd_spy:.3e}")
    print(f"[vol stats] sp500 p5/50/95 = {np.percentile(vol_sp,5):.3f}/{np.percentile(vol_sp,50):.3f}/{np.percentile(vol_sp,95):.3f}")

    # Subsample windows for training speed (still large = real data)
    rng = np.random.default_rng(0)
    sub_n = 12000
    idx_sub = rng.choice(X_sp_n.shape[0], size=min(sub_n, X_sp_n.shape[0]), replace=False)
    X_sp_train, vol_sp_train = X_sp_n[idx_sub], vol_sp[idx_sub]

    # SPY (smaller, use all)
    X_spy_train, vol_spy_train = X_spy_n, vol_spy
    print(f"[train sizes] sp500={X_sp_train.shape}, spy={X_spy_train.shape}")

    # Evaluation: target quantiles spanning empirical 10..90 percentile of vol
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    targets_sp = np.quantile(vol_sp, qs).astype(np.float32)
    targets_spy = np.quantile(vol_spy, qs).astype(np.float32)
    print(f"[targets] sp500={targets_sp}, spy={targets_spy}")

    all_results = {"sp500": {}, "spy": {}}
    for ds_name, X_train, vol_train, targets, mu_, sd_, real_pool in [
        ("sp500", X_sp_train, vol_sp_train, targets_sp, mu_sp, sd_sp, X_sp),
        ("spy", X_spy_train, vol_spy_train, targets_spy, mu_spy, sd_spy, X_spy),
    ]:
        print(f"\n========= dataset: {ds_name} =========")
        per_seed = {}
        for seed in args.seeds:
            print(f"\n--- seed={seed} on {ds_name} ---")
            seed_res = {}

            # ---- proposed diffusion (cond + SF loss) ----
            t0 = time.time()
            model, sched, _ = train_diffusion(
                X_train, vol_train, variant="full",
                seed=seed, steps=args.steps, batch=256, use_cond=True, sf_weight=0.5,
            )
            print(f"  diff_full trained in {time.time()-t0:.1f}s")
            gen_fn = lambda n, w, cond=None: diffusion_generate(model, sched, n, w, cond=cond)
            r, *_ = primary_spearman("conditional", gen_fn, targets, args.n_per_q, win, mu_, sd_)
            r.update(calibration_metrics("conditional", gen_fn, targets, args.n_per_q, win, mu_, sd_, real_pool))
            seed_res["diff_full"] = r
            del model; torch.cuda.empty_cache()

            # ---- ablation: diffusion no stylized-fact loss ----
            t0 = time.time()
            model, sched, _ = train_diffusion(
                X_train, vol_train, variant="no_sf",
                seed=seed, steps=args.steps, batch=256, use_cond=True, sf_weight=0.0,
            )
            print(f"  diff_no_sf trained in {time.time()-t0:.1f}s")
            gen_fn = lambda n, w, cond=None: diffusion_generate(model, sched, n, w, cond=cond)
            r, *_ = primary_spearman("conditional", gen_fn, targets, args.n_per_q, win, mu_, sd_)
            r.update(calibration_metrics("conditional", gen_fn, targets, args.n_per_q, win, mu_, sd_, real_pool))
            seed_res["diff_no_sf"] = r
            del model; torch.cuda.empty_cache()

            # ---- ablation: diffusion shuffled vol labels ----
            t0 = time.time()
            model, sched, _ = train_diffusion(
                X_train, vol_train, variant="shuffled",
                seed=seed, steps=args.steps, batch=256, use_cond=True, sf_weight=0.5,
                shuffled_vol=True,
            )
            print(f"  diff_shuffled trained in {time.time()-t0:.1f}s")
            gen_fn = lambda n, w, cond=None: diffusion_generate(model, sched, n, w, cond=cond)
            r, *_ = primary_spearman("conditional", gen_fn, targets, args.n_per_q, win, mu_, sd_)
            r.update(calibration_metrics("conditional", gen_fn, targets, args.n_per_q, win, mu_, sd_, real_pool))
            seed_res["diff_shuffled"] = r
            del model; torch.cuda.empty_cache()

            # ---- ablation: unconditional diffusion (no risk control) ----
            t0 = time.time()
            model, sched, _ = train_diffusion(
                X_train, vol_train, variant="unc",
                seed=seed, steps=args.steps, batch=256, use_cond=False, sf_weight=0.5,
            )
            print(f"  diff_unc trained in {time.time()-t0:.1f}s")
            gen_fn = lambda n, w, cond=None: diffusion_generate(model, sched, n, w, cond=None)
            r, *_ = primary_spearman("unconditional", gen_fn, targets, args.n_per_q, win, mu_, sd_)
            r.update(calibration_metrics("unconditional", gen_fn, targets, args.n_per_q, win, mu_, sd_, real_pool))
            seed_res["diff_unc"] = r
            del model; torch.cuda.empty_cache()

            # ---- baseline: TimeGAN-lite (unconditional + post-hoc binning) ----
            t0 = time.time()
            G = train_timegan_lite(X_train, seed=seed, steps=args.gan_steps, batch=256, cond_dim=0, V=None)
            print(f"  timegan trained in {time.time()-t0:.1f}s")
            gen_fn = lambda n, w, cond=None: gan_generate(G, n, w, cond=None)
            r, *_ = primary_spearman("unconditional", gen_fn, targets, args.n_per_q, win, mu_, sd_)
            r.update(calibration_metrics("unconditional", gen_fn, targets, args.n_per_q, win, mu_, sd_, real_pool))
            seed_res["timegan"] = r
            del G; torch.cuda.empty_cache()

            # ---- baseline: C-RNN-GAN (unconditional + post-hoc binning) ----
            t0 = time.time()
            G = train_crnn_gan(X_train, seed=seed, steps=args.gan_steps, batch=256)
            print(f"  crnn_gan trained in {time.time()-t0:.1f}s")
            gen_fn = lambda n, w, cond=None: gan_generate(G, n, w, cond=None)
            r, *_ = primary_spearman("unconditional", gen_fn, targets, args.n_per_q, win, mu_, sd_)
            r.update(calibration_metrics("unconditional", gen_fn, targets, args.n_per_q, win, mu_, sd_, real_pool))
            seed_res["crnn_gan"] = r
            del G; torch.cuda.empty_cache()

            # ---- baseline: Sig-W-GAN-lite (conditional) ----
            t0 = time.time()
            G = train_sigw_gan(X_train, vol_train, seed=seed, steps=args.gan_steps, batch=256)
            print(f"  sigw_gan trained in {time.time()-t0:.1f}s")
            gen_fn = lambda n, w, cond=None: gan_generate(G, n, w, cond=cond)
            r, *_ = primary_spearman("conditional", gen_fn, targets, args.n_per_q, win, mu_, sd_)
            r.update(calibration_metrics("conditional", gen_fn, targets, args.n_per_q, win, mu_, sd_, real_pool))
            seed_res["sigw_gan"] = r
            del G; torch.cuda.empty_cache()

            per_seed[str(seed)] = seed_res
            # save partial progress after every seed (defensive against crashes)
            all_results[ds_name] = per_seed
            with open(os.path.join(OUT, "raw_results.json"), "w") as f:
                json.dump(all_results, f, indent=2)
        all_results[ds_name] = per_seed

    with open(os.path.join(OUT, "raw_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n[done] wrote raw_results.json")


if __name__ == "__main__":
    main()
