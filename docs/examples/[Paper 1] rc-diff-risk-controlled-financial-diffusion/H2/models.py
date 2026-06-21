"""Models: conditional 1D diffusion (proposed + ablations), TimeGAN-lite, C-RNN-GAN-lite, Sig-W-GAN-lite."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock1D(nn.Module):
    def __init__(self, ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, ch)

    def forward(self, x, e):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(e))[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class CondDiffusionUNet(nn.Module):
    """Compact 1D U-Net for return-series diffusion, conditioned on time-step embedding and a continuous control (target vol)."""
    def __init__(self, win=64, ch=64, emb_dim=128, use_cond=True):
        super().__init__()
        self.use_cond = use_cond
        self.win = win
        self.ch = ch
        self.in_proj = nn.Conv1d(1, ch, 3, padding=1)
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        if use_cond:
            self.c_mlp = nn.Sequential(nn.Linear(1, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.down1 = ResBlock1D(ch, emb_dim)
        self.down2 = ResBlock1D(ch, emb_dim)
        self.pool = nn.AvgPool1d(2)
        self.mid_proj = nn.Conv1d(ch, ch * 2, 1)
        self.mid = ResBlock1D(ch * 2, emb_dim)
        self.mid_back = nn.Conv1d(ch * 2, ch, 1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.up1 = ResBlock1D(ch, emb_dim)
        self.up2 = ResBlock1D(ch, emb_dim)
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_proj = nn.Conv1d(ch, 1, 3, padding=1)

    def forward(self, x, t, cond=None):
        # x: (B, win)
        x = x.unsqueeze(1)  # (B, 1, win)
        t_emb = sinusoidal_embedding(t, 128)
        e = self.t_mlp(t_emb)
        if self.use_cond:
            assert cond is not None
            c = self.c_mlp(cond.float().view(-1, 1))
            e = e + c
        h = self.in_proj(x)
        h1 = self.down1(h, e)
        h2 = self.down2(h1, e)
        hd = self.pool(h2)
        hm = self.mid(self.mid_proj(hd), e)
        hm = self.mid_back(hm)
        hu = self.up(hm)
        # pad if mismatched
        if hu.size(-1) != h2.size(-1):
            hu = F.pad(hu, (0, h2.size(-1) - hu.size(-1)))
        u = self.up1(hu + h2, e)
        u = self.up2(u + h1, e)
        out = self.out_proj(F.silu(self.out_norm(u)))
        return out.squeeze(1)


class DiffusionScheduler:
    def __init__(self, T=200, device="cuda"):
        self.T = T
        betas = torch.linspace(1e-4, 0.02, T, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_ab = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_ab = torch.sqrt(1 - self.alpha_bars)

    def q_sample(self, x0, t, noise):
        sa = self.sqrt_ab[t].unsqueeze(-1)
        so = self.sqrt_one_minus_ab[t].unsqueeze(-1)
        return sa * x0 + so * noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t_idx, cond=None):
        beta = self.betas[t_idx]
        alpha = self.alphas[t_idx]
        ab = self.alpha_bars[t_idx]
        eps = model(x_t, torch.full((x_t.size(0),), t_idx, device=x_t.device, dtype=torch.long), cond=cond)
        mean = (1.0 / torch.sqrt(alpha)) * (x_t - beta / torch.sqrt(1 - ab) * eps)
        if t_idx > 0:
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(beta) * noise
        return mean

    @torch.no_grad()
    def sample(self, model, n, win, device, cond=None):
        x = torch.randn(n, win, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, cond=cond)
        return x


# ---------------- TimeGAN-lite (compact) ----------------

class GRUGen(nn.Module):
    def __init__(self, latent=16, hidden=64, win=64, cond_dim=0):
        super().__init__()
        self.latent = latent
        self.win = win
        self.cond_dim = cond_dim
        self.cell = nn.GRU(latent + cond_dim, hidden, batch_first=True, num_layers=2)
        self.out = nn.Linear(hidden, 1)

    def forward(self, B, device, cond=None):
        z = torch.randn(B, self.win, self.latent, device=device)
        if self.cond_dim > 0:
            c = cond.float().view(B, 1, 1).expand(B, self.win, self.cond_dim)
            z = torch.cat([z, c], dim=-1)
        h, _ = self.cell(z)
        return self.out(h).squeeze(-1)


class GRUDisc(nn.Module):
    def __init__(self, hidden=64, win=64, cond_dim=0):
        super().__init__()
        self.cond_dim = cond_dim
        self.cell = nn.GRU(1 + cond_dim, hidden, batch_first=True, num_layers=2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, cond=None):
        h = x.unsqueeze(-1)
        if self.cond_dim > 0:
            c = cond.float().view(-1, 1, 1).expand(-1, x.size(1), self.cond_dim)
            h = torch.cat([h, c], dim=-1)
        h, _ = self.cell(h)
        return self.head(h[:, -1, :]).squeeze(-1)


# ---------------- C-RNN-GAN-lite ----------------
# Mogren-style: continuous LSTM generator with adversarial loss on full sequence
class LSTMGen(nn.Module):
    def __init__(self, latent=8, hidden=64, win=64):
        super().__init__()
        self.win = win
        self.latent = latent
        self.cell = nn.LSTM(latent, hidden, batch_first=True, num_layers=2)
        self.out = nn.Linear(hidden, 1)

    def forward(self, B, device):
        z = torch.randn(B, self.win, self.latent, device=device)
        h, _ = self.cell(z)
        return self.out(h).squeeze(-1)


class LSTMDisc(nn.Module):
    def __init__(self, hidden=64, win=64):
        super().__init__()
        self.cell = nn.LSTM(1, hidden, batch_first=True, num_layers=2, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        h, _ = self.cell(x.unsqueeze(-1))
        return self.head(h.mean(dim=1)).squeeze(-1)
