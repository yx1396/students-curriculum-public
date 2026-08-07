import numpy as np
from scipy import stats


def ols_inference(x, y) -> dict:
    """一元线性回归 y = b0 + b1 x 的推断：返回 b0,b1,se1,t1,p1（双侧）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)

    return {
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "se1": float(se[1]),
        "t1": float(t[1]),
        "p1": float(p[1])
    }
