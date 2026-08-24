"""nanograd L4：numpy 张量自动微分（+,*,@,sum,relu），支持广播的反向传播。

学生任务：实现下列运算的前向传播和反向传播。
每个运算都需要正确处理广播（使用 _unbroadcast 辅助函数）。
"""
import numpy as np


def _unbroadcast(grad, shape):
    """把 grad 求和回 shape（广播的逆操作）。

    参数：
        grad: 要还原的梯度数组
        shape: 目标形状

    返回：
        缩减后与 shape 匹配的梯度
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        """初始化张量。

        参数：
            data: 数组数据（会转换为 float64 的 numpy 数组）
            _children: 产生该张量的父张量（用于构建计算图）
            _op: 产生该张量的操作名称（调试用）
        """
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        """逐元素加法，支持广播。

        提示：反向传播时需要用 _unbroadcast 把梯度缩减回 self 和 other 各自的形状。
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad = self.grad + _unbroadcast(out.grad, self.data.shape)
            other.grad = other.grad + _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        """逐元素乘法，支持广播。

        提示：乘法的反向传播：对 self 的梯度是 other.data * out.grad，对 other 的梯度是 self.data * out.grad。
        同样需要用 _unbroadcast 处理广播。
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad = self.grad + _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad = other.grad + _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        """矩阵乘法。

        提示：self (M,K) @ other (K,N) → out (M,N)
        d_self = out.grad @ other.data.T
        d_other = self.data.T @ out.grad
        """
        out = Tensor(self.data @ other.data, (self, other), "@")
        def _backward():
            self.grad = self.grad + out.grad @ other.data.T
            other.grad = other.grad + self.data.T @ out.grad
        out._backward = _backward
        return out

    def sum(self):
        """对所有元素求和，返回标量张量。

        提示：反向传播时，梯度均匀分配给所有元素。
        """
        out = Tensor(self.data.sum(), (self,), "sum")
        def _backward():
            self.grad = self.grad + np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        """逐元素 ReLU 激活函数。

        提示：只有 self.data > 0 的位置才传递梯度。
        """
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")
        def _backward():
            self.grad = self.grad + (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def backward(self):
        """拓扑排序后反向传播，计算所有叶子节点的梯度。"""
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
