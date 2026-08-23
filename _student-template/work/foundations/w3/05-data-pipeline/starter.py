"""W3 数据管道小项目：实现 build_pipeline。见 projects/p-w3-data-pipeline/README.md。"""
from __future__ import annotations
import pandas as pd


def build_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """清洗合成表格数据：去重、数值列中位数填补缺失、group 统一大写、重置索引。返回清洗后的 DataFrame。"""
    out = df.drop_duplicates().copy()
    for col in out.select_dtypes(include='number').columns:
        out[col] = out[col].fillna(out[col].median())
    if 'group' in out.columns:
        out['group'] = out['group'].astype(str).str.upper()
    return out.reset_index(drop=True)
