import torch


def reparameterize(mu, logvar, seed=0):
    """重参数化技巧：z = mu + eps*std, std = exp(0.5*logvar)。"""
    torch.manual_seed(seed)
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu, logvar):
    """标准正态先验下对角高斯后验的 KL：0.5*sum(exp(logvar)+mu^2-1-logvar)（按样本求和后取均值）。"""
    kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar, dim=1)
    return kl.mean()
