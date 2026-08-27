import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """FT-Transformer 特征分词器：每个数值特征映射到一个 d_token 维 token，并前置一个 [CLS] token。"""
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))

        self.cls = nn.Parameter(torch.randn(1, d_token))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        cls = self.cls.expand(x.shape[0], 1, -1)
        return torch.cat([cls, tokens], dim=1)
