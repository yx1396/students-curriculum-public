import math


class Shape:
    def area(self) -> float:
        """计算形状的面积"""
        raise NotImplementedError


class Rectangle(Shape):
    def __init__(self, w: float, h: float):
        """初始化矩形，参数为宽度和高度"""
        self.w = w
        self.h = h

    def area(self) -> float:
        """计算矩形的面积"""
        return self.w * self.h


class Circle(Shape):
    def __init__(self, r: float):
        """初始化圆形，参数为半径"""
        self.r = r

    def area(self) -> float:
        """计算圆形的面积"""
        return math.pi * self.r ** 2


def total_area(shapes: list[Shape]) -> float:
    """计算所有形状的总面积"""
    total = 0.0

    for shape in shapes:
        total += shape.area()

    return total
