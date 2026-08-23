import numpy as np


def describe(x) -> dict:
    """返回均值、方差(总体, ddof=0)、标准差、中位数。"""
    x = np.asarray(x, dtype=float)
    return{
        "mean": x.mean(),
        "var": x.var(),
        "std": x.std(),
        "median": np.median(x)
    }
