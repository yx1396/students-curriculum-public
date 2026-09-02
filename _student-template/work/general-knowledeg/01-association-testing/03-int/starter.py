import numpy as np
from scipy import stats


def rank_int(x, offset=0.5):
    """基于秩的逆正态变换 (rank-based inverse-normal transform)，把 x 近似变成标准正态。"""
    x = np.array(x, dtype=float)
    ranks = stats.rankdata(x)
    n = len(x)
    q = (ranks - offset) / (n - 2 * offset + 1)
    return stats.norm.ppf(q)
