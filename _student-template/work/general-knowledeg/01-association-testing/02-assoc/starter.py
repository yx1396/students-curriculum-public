import numpy as np
from scipy import stats


def assoc(G, y, covariates=None):
    """对每个 SNP 做单变量线性回归；协变量用 Frisch–Waugh–Lovell 残差化后再回归。
    G:(n,m) 基因型; y:(n,); covariates:(n,k) 或 None（截距总是包含）。
    返回 {'beta','se','t','p'}，每个长度 m。"""
    G = np.asarray(G, dtype=float)
    y = np.asarray(y, dtype=float)
    n = G.shape[0]
    C = np.ones((n, 1)) if covariates is None else np.column_stack([np.ones(n), covariates])
    Cpinv = np.linalg.pinv(C)

    def resid(M):
        """把矩阵/向量 M 对 C 做残差化。"""
        return M - C @ (Cpinv @ M)

    yr = resid(y)
    Gr = resid(G)
    dof = n - C.shape[1] - 1

    Sxx = (Gr ** 2).sum(axis=0)
    beta = (Gr * yr[:, None]).sum(axis=0) / Sxx
    resid_y = yr[:, None] - Gr * beta[None, :]
    sigma2 = (resid_y ** 2).sum(axis=0) / dof
    se = np.sqrt(sigma2 / Sxx)
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return {"beta": beta, "se": se, "t": t, "p": p}
