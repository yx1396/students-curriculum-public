import numpy as np


def standardize_columns(X: np.ndarray) -> np.ndarray:
    """对每一列做 z-score（用广播），返回同形状数组。"""
    mu = X.mean(axis=0)
    st = X.std(axis=0)
    return (X - mu) / st
