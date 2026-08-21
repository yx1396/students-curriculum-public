import torch


def grad_of_square_sum(x):
    """给定 1D 张量 x，计算 y = sum(x**2) 对 x 的梯度（返回张量，应等于 2x）。"""
    x = torch.as_tensor(x, dtype=torch.float32).requires_grad_(True)
    y = (x**2).sum()
    y.backward()
    return x.grad
