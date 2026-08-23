"""W2 NumPy 练习。实现下列函数，仅用 numpy 向量化操作（禁止显式 Python for 循环）。
把本文件复制到 progress/<你的目录>/work/w2-numpy/solution.py 后实现，再运行自动评分。"""
import numpy as np


def standardize(x: np.ndarray) -> np.ndarray:
    """对 1D 数组做 z-score 标准化：(x - mean) / std。"""
    mu = np.mean(x)
    sigma = np.std(x)
    return (x - mu) / sigma


def pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """给定 1D 数组 a (n,) 与 b (m,)，用广播返回 (n, m) 的逐点差的平方矩阵。"""
    return (a[:, None] - b) ** 2
