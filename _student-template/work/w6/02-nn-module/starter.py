import torch
import torch.nn as nn


class MLP(nn.Module):
    """两层 MLP：Linear(in,hidden)->ReLU->Linear(hidden,out)。"""
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out)
        )

    def forward(self, x):
        return self.net(x)
