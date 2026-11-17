import numpy as np

from nn import Linear
from optim import sgd_update
from tensor import Tensor


def test_linear_output_shape_and_parameters():
    layer = Linear(3, 5, rng=np.random.default_rng(0))
    out = layer(Tensor(np.ones((4, 3))))

    assert out.shape == (4, 5)
    assert [p.shape for p in layer.parameters()] == [(3, 5), (5,)]


def test_linear_bias_starts_at_zero_and_weights_use_he_scale():
    layer = Linear(400, 300, rng=np.random.default_rng(1))

    np.testing.assert_array_equal(layer.b.data, np.zeros(300))
    np.testing.assert_allclose(layer.W.data.std(), np.sqrt(2.0 / 400), rtol=0.02)


def test_from_arrays_copies_its_inputs():
    W = np.ones((2, 2))
    b = np.zeros(2)
    layer = Linear.from_arrays(W, b)
    layer.W.data -= 1.0

    np.testing.assert_array_equal(W, np.ones((2, 2)))


def test_sgd_update_reduces_loss():
    rng = np.random.default_rng(2)
    layer = Linear(3, 1, rng=rng)
    x = Tensor(rng.normal(size=(8, 3)))
    targets = rng.normal(size=(8, 1))

    def loss():
        return ((layer(x) - targets) ** 2).mean()

    before = loss()
    before.backward()
    sgd_update(layer.parameters(), learning_rate=0.1)

    assert loss().data < before.data
