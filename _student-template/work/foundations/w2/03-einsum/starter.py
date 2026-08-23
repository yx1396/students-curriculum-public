import numpy as np


def gram(X: np.ndarray) -> np.ndarray:
    """返回 Gram 矩阵 X @ X.T，用 einsum。"""
    return np.einsum("ik,jk->ij", X, X)
