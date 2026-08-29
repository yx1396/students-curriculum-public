from __future__ import annotations
from pathlib import Path
import torch


def save_checkpoint(model, optimizer, step, path):
    """保存模型/优化器状态与步数到 path。"""
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }, path)


def load_checkpoint(model, optimizer, path):
    """从 path 恢复模型/优化器状态，返回 step。"""
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]