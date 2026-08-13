import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


def fit_xgb_auc(Xtr, ytr, Xte, yte) -> float:
    """训练 XGBoost 分类器（hist），返回测试集 ROC-AUC。"""
    clf = XGBClassifier(n_estimators=50, max_depth=3, tree_method="hist",
                        eval_metric="logloss", random_state=0)
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))