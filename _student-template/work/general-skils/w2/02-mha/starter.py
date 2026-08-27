import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """多头自注意力：投影 Q/K/V → 分头 → 缩放点积注意力 → 合并 → 输出投影。"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q = split_heads(self.wq(x))
        k = split_heads(self.wk(x))
        v = split_heads(self.wv(x))

        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = scores.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)
