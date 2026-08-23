import numpy as np
from sklearn.decomposition import PCA


def pca_transform(X, k: int):
    """PCA 降到 k 维，返回 (scores[n,k], explained_variance_ratio[k])。"""
    p = PCA(n_components=k)
    scores = p.fit_transform(np.asarray(X, dtype=float))
    return scores, p.explained_variance_ratio_