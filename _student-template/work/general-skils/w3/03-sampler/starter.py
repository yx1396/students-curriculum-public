import torch


def euler_step(x, v, dt):
    """ODE 欧拉积分一步：x_{t+dt} = x + v * dt。"""
    return x + v * dt


def integrate(x0, velocity_fn, n_steps):
    """从 x0 出发，用常速度场 velocity_fn(x,t) 在 [0,1] 上欧拉积分 n_steps 步。"""
    x = x0
    dt = 1 / n_steps
    for i in range(n_steps):
        t = i * dt
        x = euler_step(x, velocity_fn(x, t), dt)
    return x

