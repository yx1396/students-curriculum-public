import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """清洗：去重、数值列用中位数填补缺失、把 group 统一为大写。返回新 DataFrame。"""
    out = df.drop_duplicates().copy()
    for col in out.select_dtypes(include='number').columns:
        out[col] = out[col].fillna(out[col].median())
    if 'group' in out.columns:
        out['group'] = out['group'].astype(str).str.upper()

    return out.reset_index(drop=True)

