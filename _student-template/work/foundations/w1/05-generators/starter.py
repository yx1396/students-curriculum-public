from __future__ import annotations
from typing import Iterable, Iterator


def moving_sum(xs: Iterable[float], k: int) -> Iterator[float]:
    """计算移动窗口和的生成器"""
    window: list[float] = []
    for x in xs:
        window.append(x)
        if len(window) == k:
            yield sum(window)
            window.pop(0)


def take(it: Iterator, n: int) -> list:
    """从迭代器中取前 n 个元素"""
    out = []
    for i, v in enumerate(it):
        if i >= n:
            break
        out.append(v)
    return out
