from __future__ import annotations
from functools import wraps
from contextlib import contextmanager


def memoize(fn):
    """装饰器，缓存函数的计算结果"""
    cache = {}
    @wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(kwargs.items()))

        if key in cache:
            return cache[key]

        result = fn(*args, **kwargs)
        cache[key] = result

        return result

    return wrapper

@contextmanager
def tag(name: str, log: list):
    """上下文管理器，记录代码块的开始和结束"""
    log.append(f"start:{name}")

    try:
        yield

    finally:
        log.append(f"end:{name}")

