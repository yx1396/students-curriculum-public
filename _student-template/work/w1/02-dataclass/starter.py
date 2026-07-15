from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Point:
    """表示二维平面上的一个不可变点"""
    x: float
    y: float

    def dist_to(self, other: "Point") -> float:
        """计算到另一个点的距离"""
        return math.hypot(self.x - other.x, self.y - other.y)

    @property
    def norm(self) -> float:
        """计算点到原点的距离（范数）"""
        return math.hypot(self.x, self.y)
