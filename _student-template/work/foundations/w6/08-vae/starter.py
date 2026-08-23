"""W6 VAE 阶段关卡：从零实现变分自编码器。见 projects/p-w6-vae/README.md。

架构说明：
  编码器（Encoder）：输入 x → 隐层 → (mu, logvar)，输出潜变量分布参数。
  重参数化（Reparameterize）：z = mu + eps * std，eps ~ N(0,1)，使梯度可回传。
  解码器（Decoder）：z → 重构 x_hat。
  损失函数 ELBO = 重构损失（MSE）+ β * KL 散度（KL = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)）。
"""
import torch
import torch.nn as nn


class VAE(nn.Module):
    """变分自编码器。

    Args:
        n_in (int): 输入维度。
        n_latent (int): 潜空间维度。
    """

    def __init__(self, n_in, n_latent):
        super().__init__()
        # TODO: 定义编码器（enc）、均值层（mu）、对数方差层（logvar）、解码器（dec）
        # Encoder: x -> hidden
        features = 8
        self.enc = nn.Sequential(
            nn.Linear(n_in, features),
            nn.ReLU(),
        )

        # hidden -> mu
        self.mu = nn.Linear(features, n_latent)

        # hidden -> logvar
        self.logvar = nn.Linear(features, n_latent)

        # Decoder: z -> x_hat
        self.dec = nn.Sequential(
            nn.Linear(n_latent, features),
            nn.ReLU(),
            nn.Linear(features, n_in)
        )


    def encode(self, x):
            """前向编码：x → (mu, logvar)。"""
            h = self.enc(x)

            mu = self.mu(h)

            logvar = self.logvar(h)

            return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化：z = mu + eps * std，std = exp(0.5 * logvar)。"""
        std = torch.exp(0.5 * logvar)

        # eps ~ N(0,1)
        eps = torch.randn_like(std)

        # z = mu + eps * std
        z = mu + eps * std

        return z

    def forward(self, x):
        """完整前向传播：x → (x_hat, mu, logvar)。"""
        # 编码
        mu, logvar = self.encode(x)

        # 采样潜变量
        z = self.reparameterize(mu, logvar)

        # 解码
        x_hat = self.dec(z)

        return x_hat, mu, logvar


def build_vae(n_in, n_latent):
    """构建并返回 VAE 模型实例。"""
    model = VAE(n_in, n_latent)

    return model


def elbo_loss(x_hat, x, mu, logvar, beta=1.0):
    """计算 ELBO 损失：recon_loss + beta * kl_loss。

    recon_loss = mean over samples of sum((x_hat - x)^2 over features)
    kl_loss    = mean over samples of 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
    """
    # 每个样本所有feature求和
    recon_loss = torch.sum(
        (x_hat - x) ** 2,
        dim=1
    ).mean()

    # =====================
    # KL散度
    # =====================

    kl_loss = 0.5 * torch.sum(
        torch.exp(logvar)
        + mu ** 2
        - 1
        - logvar,
        dim=1
    ).mean()

    # 总loss
    loss = recon_loss + beta * kl_loss

    return loss


def train_vae(X, epochs=300, lr=1e-2, seed=0):
    """在数据矩阵 X (torch.Tensor 或可转张量) 上训练 VAE，返回 (model, final_loss)。

    使用 Adam 优化器，固定随机种子以保证可复现。
    """
    # 固定随机种子
    torch.manual_seed(seed)

    # 转Tensor
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(
            X,
            dtype=torch.float32
        )
    else:
        X = X.float()

    # 输入维度
    n_in = X.shape[1]

    # 默认latent维度
    n_latent = 2

    # 创建模型
    model = build_vae(
        n_in,
        n_latent
    )

    # Adam优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    model.train()

    final_loss = None

    for epoch in range(epochs):
        optimizer.zero_grad()

        # forward
        x_hat, mu, logvar = model(X)

        # loss
        loss = elbo_loss(
            x_hat,
            X,
            mu,
            logvar
        )

        # backward
        loss.backward()

        # 更新参数
        optimizer.step()

        final_loss = loss.item()

    return model, final_loss
