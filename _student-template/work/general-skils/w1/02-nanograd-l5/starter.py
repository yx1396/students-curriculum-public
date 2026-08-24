"""nanograd L5：numpy 张量自动微分 + Module/Linear/SGD。

学生任务：在已提供的 Tensor 类基础上，实现 Module、Linear 和 SGD。
Tensor 类已完整实现，请勿修改。
"""
import numpy as np


def _unbroadcast(grad, shape):
    """把 grad 求和回 shape（广播的逆操作）。"""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """numpy 张量，支持自动微分。（已完整实现，请勿修改）"""

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad = self.grad + _unbroadcast(out.grad, self.data.shape)
            other.grad = other.grad + _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = self.grad + _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad = other.grad + _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad = self.grad + out.grad @ other.data.T
            other.grad = other.grad + self.data.T @ out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad = self.grad + np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def mean(self):
        n = self.data.size
        out = Tensor(self.data.mean(), (self,), "mean")

        def _backward():
            self.grad = self.grad + np.ones_like(self.data) * out.grad / n
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad = self.grad + (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def backward(self):
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


class Module:
    """神经网络模块基类。

    学生任务：实现 parameters() 和 zero_grad()。
    - parameters() 返回该模块所有可学习参数的列表（默认返回空列表）。
    - zero_grad() 将所有参数的梯度清零。
    """

    def parameters(self):
        """返回所有可学习参数（Tensor 列表）。"""
        return []

    def zero_grad(self):
        """将所有参数的梯度归零。"""
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


class Linear(Module):
    """全连接线性层：out = x @ w + b。

    学生任务：实现 __init__、__call__ 和 parameters()。
    - __init__(nin, nout, seed): 用正态分布初始化权重矩阵 w (nin, nout)，偏置 b (nout,) 初始化为 0。
      权重标准差为 1/sqrt(nin)，使用 np.random.default_rng(seed) 生成随机数。
    - __call__(x): 计算 x @ w + b（x 是 Tensor）。
    - parameters(): 返回 [self.w, self.b]。
    """

    def __init__(self, nin, nout, seed=0):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(nin)
        self.w = Tensor(rng.normal(0, scale, (nin, nout)))
        self.b = Tensor(np.zeros(nout))

    def __call__(self, x):
        return x @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]


class SGD:
    """随机梯度下降优化器。

    学生任务：实现 __init__、zero_grad() 和 step()。
    - __init__(params, lr): 保存参数列表和学习率。
    - zero_grad(): 将所有参数的梯度清零（p.grad = zeros_like(p.data)）。
    - step(): 执行一步梯度下降：p.data -= lr * p.grad。
    """

    def __init__(self, params, lr=0.1):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad
