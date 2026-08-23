"""nanograd L3：从零训练 MLP 在 XOR 上。

本文件是 L3 的学生入门框架。
学生需要实现 Neuron、Layer、MLP 类和 train_xor 函数。
"""

import math
import random


class Value:
    """一个标量值，支持自动微分和更多运算。"""

    def __init__(self, data, _children=(), _op=""):
        """初始化一个 Value 对象。

        Args:
            data: 标量数值
            _children: 子节点（默认为空）
            _op: 操作类型（默认为空字符串）
        """
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float))
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += (k * self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


class Neuron:
    """一个神经元，包含权重和偏置。

    学生需要实现：
    - __init__: 初始化权重和偏置
    - __call__: 前向传播
    - parameters: 返回参数列表
    """

    def __init__(self, nin, seed=0):
        """初始化一个神经元。

        Args:
            nin: 输入维数
            seed: 随机种子
        """
        rng = random.Random(seed)
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        """前向传播。

        Args:
            x: 输入向量

        Returns:
            tanh 激活后的输出
        """
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        """返回所有参数（权重 + 偏置）。"""
        return self.w + [self.b]


class Layer:
    """一个神经网络层，包含多个神经元。

    学生需要实现：
    - __init__: 初始化神经元
    - __call__: 前向传播
    - parameters: 返回所有参数
    """

    def __init__(self, nin, nout, seed=0):
        """初始化一个层。

        Args:
            nin: 输入维数
            nout: 输出维数
            seed: 随机种子
        """
        self.neurons = [Neuron(nin, seed + i) for i in range(nout)]

    def __call__(self, x):
        """前向传播。

        Args:
            x: 输入向量

        Returns:
            单个输出或输出列表
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """返回所有参数。"""
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """多层感知机。

    学生需要实现：
    - __init__: 初始化各层
    - __call__: 前向传播
    - parameters: 返回所有参数
    """

    def __init__(self, nin, nouts, seed=0):
        """初始化 MLP。

        Args:
            nin: 输入维数
            nouts: 各层输出维数列表
            seed: 随机种子
        """
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1], seed + 10 * i)
                       for i in range(len(nouts))]

    def __call__(self, x):
        """前向传播。

        Args:
            x: 输入向量

        Returns:
            输出
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """返回所有参数。"""
        return [p for layer in self.layers for p in layer.parameters()]


def train_xor(steps=200, lr=0.1, seed=1):
    """在 XOR 上训练一个小 MLP，返回 (model, final_loss)。

    Args:
        steps: 训练步数
        lr: 学习率
        seed: 随机种子

    Returns:
        (model, final_loss)
    """
    XOR_DATA = [
        ([0.0, 0.0], -1.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], -1.0),
    ]

    model = MLP(2, [4, 1], seed=seed)

    for step in range(steps):

        # 清零梯度
        for p in model.parameters():
            p.grad = 0.0

        # forward + loss
        loss = Value(0.0)

        for x, y in XOR_DATA:
            pred = model([Value(v) for v in x])
            loss += (pred - y) ** 2

        # backward
        loss.backward()

        # SGD
        for p in model.parameters():
            p.data -= lr * p.grad

    return model, loss.data