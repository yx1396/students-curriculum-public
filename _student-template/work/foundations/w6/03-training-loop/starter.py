import torch
import torch.nn as nn


def train_linear(steps=300, lr=0.1, seed=0):
    """在 y=2x+1 的合成数据上训练一个 nn.Linear(1,1)，返回 (model, final_loss)。"""
    torch.manual_seed(seed)
    x = torch.linspace(-1, 1, 64).unsqueeze(1)
    y = 2 * x + 1

    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss = None
    for step in range(steps):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

    return model, float(loss)
