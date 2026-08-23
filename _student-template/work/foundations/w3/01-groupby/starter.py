import pandas as pd


def group_means(df: pd.DataFrame, key: str, value: str) -> pd.DataFrame:
    """按 key 分组求 value 的均值，返回两列 DataFrame（key, value），按 key 升序。"""
    return df.groupby(key)[value].mean().reset_index()
