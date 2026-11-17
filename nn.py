"""Neural network layers built on Tensor."""

import numpy as np

from tensor import Tensor


class Linear:
    """Fully connected layer computing x @ W + b."""

    def __init__(self, in_features, out_features, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        # He initialization keeps activations at a stable scale through ReLU layers.
        scale = np.sqrt(2.0 / in_features)
        self.W = Tensor(rng.normal(0.0, scale, size=(in_features, out_features)))
        self.b = Tensor(np.zeros(out_features))

    @classmethod
    def from_arrays(cls, W, b):
        """Build a layer with given weights, e.g. to match another implementation."""
        layer = cls.__new__(cls)
        layer.W = Tensor(W)
        layer.b = Tensor(b)
        return layer

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]
