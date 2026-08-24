"""nanograd L6：numpy 张量自动微分 + Module/Linear/SGD + 融合 softmax 交叉熵。

学生任务：在已提供的 Tensor/Linear/SGD 基础上，实现 cross_entropy 函数。
Tensor、Module、Linear、SGD 均已完整实现，请勿修改。
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
    """神经网络模块基类。（已完整实现，请勿修改）"""

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


class Linear(Module):
    """全连接线性层：out = x @ w + b。（已完整实现，请勿修改）"""

    def __init__(self, nin, nout, seed=0):
        rng = np.random.default_rng(seed)
        self.w = Tensor(rng.normal(0, 1.0 / np.sqrt(nin), (nin, nout)))
        self.b = Tensor(np.zeros(nout))

    def __call__(self, x):
        return x @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]


class SGD:
    """随机梯度下降优化器。（已完整实现，请勿修改）"""

    def __init__(self, params, lr=0.1):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self):
        for p in self.params:
            p.data = p.data - self.lr * p.grad


def cross_entropy(logits, targets):
    """融合 softmax 的交叉熵损失函数。

    参数：
        logits: Tensor，形状 (B, C)，未经 softmax 的原始分数
        targets: int 数组，形状 (B,)，每个样本的真实类别索引

    返回：
        标量 Tensor，交叉熵损失（平均值）

    提示（数值稳定）：
        1. 减去每行最大值：z = logits.data - logits.data.max(axis=1, keepdims=True)
        2. 计算 softmax：sm = exp(z) / exp(z).sum(axis=1, keepdims=True)
        3. 损失 = -log(sm[arange(B), targets]).mean()
        4. 梯度：dL/dlogits = (softmax - onehot) / B，只在 backward 中修改 logits.grad
    """
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    exp = np.exp(z)
    sm = exp / exp.sum(axis=1, keepdims=True)

    B = z.shape[0]
    targets = np.asarray(targets, dtype=int)

    loss_val = float(-np.log(sm[np.arange(B), targets] + 1e-12).mean())
    out = Tensor(loss_val, (logits,), "ce")

    def _backward():
        grad = sm.copy()
        grad[np.arange(B), targets] -= 1.0
        grad /= B
        logits.grad = logits.grad + grad * out.grad

    out._backward = _backward
    return out

