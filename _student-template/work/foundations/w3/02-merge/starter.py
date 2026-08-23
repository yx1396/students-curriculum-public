import pandas as pd


def join_orders(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """按 customer_id 做 inner join，返回合并后的表。"""
    return customers.merge(orders, on='customer_id', how='inner')
