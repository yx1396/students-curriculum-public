import numpy as np


def kfold_indices(n: int, k: int, seed: int = 0):
    """返回 k 折的 (train_idx, test_idx) 列表，测试折两两不相交且并集为全部样本。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    floads = np.array_split(idx, k)

    out = []

    for i in range(k):
        test = np.sort(floads[i])
        train = np.sort(np.concatenate([floads[j] for j in range(k) if j != i]))
        out.append((train, test))

    return out
