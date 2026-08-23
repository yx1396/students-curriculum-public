import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """返回 auc、accuracy 及混淆矩阵四格 (tp,fp,tn,fn)。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tn + fp + fn + tp
    return{
        "auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float((tp + tn) / total),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

