import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_predict_proba(Xtr, ytr, Xte):
    """拟合逻辑回归，返回 Xte 的正类预测概率（一维数组）。"""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]
