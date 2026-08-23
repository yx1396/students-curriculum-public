import numpy as np


def bonferroni(pvals, alpha: float = 0.05):
    """Bonferroni 校正：返回布尔数组（是否拒绝原假设）。"""
    p = np.asarray(pvals, dtype=float)
    return p < (alpha / len(p))


def bh_fdr(pvals, alpha: float = 0.05):
    """Benjamini–Hochberg FDR：返回布尔数组（保持原顺序）。"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    thresh = alpha * (np.arange(1, n + 1) / n)
    below = ranked <= thresh

    reject = np.zeros(n, dtype=bool)

    if below.any():
        kmax = int(np.max(np.where(below)[0]))
        reject[order[:kmax + 1]] = True
    return reject
