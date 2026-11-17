"""Reverse-mode automatic differentiation on NumPy arrays."""

import numpy as np


def unbroadcast(grad, shape):
    """Sum grad down to shape, undoing NumPy broadcasting from the forward pass."""
    # Broadcasting may have prepended dimensions; sum them away.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Dimensions of size 1 may have been stretched; sum them back to size 1.
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


def as_tensor(value):
    return value if isinstance(value, Tensor) else Tensor(value)


class Tensor:
    # Make NumPy defer to Tensor, so `array + tensor` calls Tensor.__radd__.
    __array_ufunc__ = None

    def __init__(self, data, parents=()):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = parents
        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def __repr__(self):
        return f"Tensor({self.data!r})"

    # Elementwise arithmetic.

    def __add__(self, other):
        other = as_tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def backward():
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)

        out._backward = backward
        return out

    def __mul__(self, other):
        other = as_tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def backward():
            self.grad += unbroadcast(out.grad * other.data, self.shape)
            other.grad += unbroadcast(out.grad * self.data, other.shape)

        out._backward = backward
        return out

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("Tensor exponents must be Python numbers")
        out = Tensor(self.data ** exponent, (self,))

        def backward():
            self.grad += out.grad * exponent * self.data ** (exponent - 1)

        out._backward = backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-as_tensor(other))

    def __rsub__(self, other):
        return as_tensor(other) + (-self)

    def __truediv__(self, other):
        return self * as_tensor(other) ** -1

    def __rtruediv__(self, other):
        return as_tensor(other) * self ** -1

    # Matrix multiplication.

    def __matmul__(self, other):
        other = as_tensor(other)
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("matmul supports 2-D tensors only")
        out = Tensor(self.data @ other.data, (self, other))

        def backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = backward
        return out

    def __rmatmul__(self, other):
        return as_tensor(other) @ self

    # Reductions.

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def backward():
            grad = out.grad
            if axis is not None and not keepdims:
                # Restore the reduced axes so the gradient broadcasts correctly.
                grad = np.expand_dims(grad, axis)
            self.grad += np.broadcast_to(grad, self.shape)

        out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            count = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    # Elementwise functions.

    def exp(self):
        out = Tensor(np.exp(self.data), (self,))

        def backward():
            self.grad += out.grad * out.data

        out._backward = backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,))

        def backward():
            self.grad += out.grad / self.data

        out._backward = backward
        return out

    # Activations.

    def relu(self):
        active = self.data > 0.0
        out = Tensor(self.data * active, (self,))

        def backward():
            self.grad += out.grad * active

        out._backward = backward
        return out

    def log_softmax(self, axis=-1):
        # Subtracting the max leaves log-softmax unchanged but keeps exp() from overflowing.
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        log_normalizer = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        out = Tensor(shifted - log_normalizer, (self,))

        def backward():
            softmax = np.exp(out.data)
            self.grad += out.grad - softmax * out.grad.sum(axis=axis, keepdims=True)

        out._backward = backward
        return out

    # Backpropagation.

    def backward(self):
        if self.size != 1:
            raise ValueError("backward() needs a single-element tensor such as a loss")

        # Iterative depth-first search, so deep graphs cannot overflow the call stack.
        order = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, parents_done = stack.pop()
            if parents_done:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for parent in node._parents:
                if parent not in visited:
                    stack.append((parent, False))

        # Start a fresh backward pass.
        for node in order:
            node.grad = np.zeros_like(node.data)
        self.grad = np.ones_like(self.data)

        for node in reversed(order):
            node._backward()
