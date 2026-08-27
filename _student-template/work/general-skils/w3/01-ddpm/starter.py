import torch


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """线性 beta 噪声表，返回 (betas, alphas, alpha_bars)，每个长度 T。"""
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def q_sample(x0, t, noise, alpha_bars):
    """前向加噪：x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) noise。
    x0,noise: (B, D); t: (B,) 长整型索引。"""
    ab = alpha_bars[t].unsqueeze(1)
    return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
