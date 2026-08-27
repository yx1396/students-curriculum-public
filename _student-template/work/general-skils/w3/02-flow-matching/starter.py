import torch


def ot_interpolate(x0, x1, t):
    """OT 条件流匹配的线性插值路径：x_t = (1-t) x0 + t x1。t: (B,1) 或标量。"""
    return (1.0 - t) * x0 + t * x1


def velocity_target(x0, x1):
    """线性路径的目标速度场 v = dx_t/dt = x1 - x0。"""
    return x1 - x0
