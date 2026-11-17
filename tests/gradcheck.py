"""Shared finite-difference gradient checks for the test suite."""

import numpy as np

from tensor import Tensor


def random_array(rng, shape, low=-1.0, high=1.0):
    return rng.uniform(low, high, size=shape)


def away_from_zero(rng, shape):
    """Random values with |x| >= 0.2, so ReLU kinks and division by zero are avoided."""
    magnitude = rng.uniform(0.2, 1.0, size=shape)
    sign = rng.choice([-1.0, 1.0], size=shape)
    return magnitude * sign


def numerical_gradients(fn, arrays, epsilon=1e-6):
    """Central-difference gradients of sum(fn(*arrays) * weights) for each input."""
    grads = []
    for index, array in enumerate(arrays):
        grad = np.zeros_like(array)
        for position in np.ndindex(array.shape):
            plus = [a.copy() for a in arrays]
            minus = [a.copy() for a in arrays]
            plus[index][position] += epsilon
            minus[index][position] -= epsilon
            grad[position] = (fn(*plus) - fn(*minus)) / (2.0 * epsilon)
        grads.append(grad)
    return grads


def check_gradients(op, *arrays, seed=0):
    """Compare autograd gradients of op against finite differences."""
    rng = np.random.default_rng(seed)
    # 0-d inputs come back from NumPy as scalars; make them writable arrays.
    arrays = [np.array(a, dtype=np.float64) for a in arrays]
    output_shape = op(*[Tensor(a) for a in arrays]).shape
    # Random output weights make every output element matter differently.
    weights = rng.uniform(0.5, 1.5, size=output_shape)

    def scalar_loss(*inputs):
        return float((op(*[Tensor(a) for a in inputs]).data * weights).sum())

    tensors = [Tensor(a) for a in arrays]
    (op(*tensors) * weights).sum().backward()

    expected = numerical_gradients(scalar_loss, arrays)
    for i, (tensor, grad) in enumerate(zip(tensors, expected)):
        assert tensor.grad.shape == tensor.shape
        np.testing.assert_allclose(
            tensor.grad, grad, rtol=1e-6, atol=1e-8, err_msg=f"input {i}"
        )
