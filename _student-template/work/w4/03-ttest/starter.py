import numpy as np
from scipy import stats


def two_sample_t(a, b) -> dict:
    """双样本等方差 t 检验（Student），手动实现，返回 t 与双侧 p。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    t = (a.mean() - b.mean()) / np.sqrt(sp2 * (1/na + 1/nb))
    dof = na + nb - 2
    p = 2 * stats.t.sf(abs(t), dof)
    return {'t': float(t), 'p': float(p)}
