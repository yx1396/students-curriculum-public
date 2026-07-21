import numpy as np


def ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """普通最小二乘 β = (XᵀX)⁻¹ Xᵀy，用 np.linalg.solve（不要显式求逆）。"""
    return np.linalg.solve(X.T @ X, X.T @ y)
