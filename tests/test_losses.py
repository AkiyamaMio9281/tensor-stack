import numpy as np
import pytest

from gradcheck import check_gradients, random_array
from losses import cross_entropy, one_hot
from nn import Linear
from optim import sgd_update
from tensor import Tensor


# exp and log


def test_exp_gradient():
    rng = np.random.default_rng(0)
    check_gradients(lambda a: a.exp(), random_array(rng, (3, 4)))


def test_log_gradient():
    rng = np.random.default_rng(1)
    check_gradients(lambda a: a.log(), random_array(rng, (3, 4), low=0.2, high=2.0))


# log_softmax


@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_log_softmax_gradient(axis):
    rng = np.random.default_rng(2)
    check_gradients(lambda a: a.log_softmax(axis=axis), random_array(rng, (4, 5), -3.0, 3.0))


def test_log_softmax_rows_are_log_probabilities():
    rng = np.random.default_rng(3)
    log_probs = Tensor(random_array(rng, (6, 10), -5.0, 5.0)).log_softmax(axis=1).data

    assert np.all(log_probs <= 0.0)
    np.testing.assert_allclose(np.exp(log_probs).sum(axis=1), np.ones(6))


def test_log_softmax_is_stable_for_large_logits():
    logits = Tensor(np.array([[1000.0, 1000.0, 999.0], [-1000.0, 0.0, 1000.0]]))
    log_probs = logits.log_softmax(axis=1)
    log_probs.sum().backward()

    assert np.all(np.isfinite(log_probs.data))
    assert np.all(np.isfinite(logits.grad))
    expected_first_row = np.array([0.0, 0.0, -1.0]) - np.log(2.0 + np.exp(-1.0))
    np.testing.assert_allclose(log_probs.data[0], expected_first_row)


def test_log_softmax_matches_pytorch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    logits_array = random_array(rng, (8, 10), -4.0, 4.0)
    weights = random_array(rng, (8, 10))

    logits = Tensor(logits_array)
    (logits.log_softmax(axis=1) * weights).sum().backward()

    torch_logits = torch.tensor(logits_array, requires_grad=True)
    torch_out = torch.nn.functional.log_softmax(torch_logits, dim=1)
    (torch_out * torch.tensor(weights)).sum().backward()

    np.testing.assert_allclose(
        logits.log_softmax(axis=1).data, torch_out.detach().numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(logits.grad, torch_logits.grad.numpy(), rtol=1e-10, atol=1e-12)


# cross_entropy


def test_one_hot():
    np.testing.assert_array_equal(
        one_hot(np.array([2, 0, 1]), 3),
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )


def test_cross_entropy_of_uniform_logits_is_log_num_classes():
    loss = cross_entropy(Tensor(np.zeros((5, 10))), np.arange(5))
    np.testing.assert_allclose(loss.data, np.log(10.0))


def test_cross_entropy_gradient():
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 7, size=6)
    check_gradients(lambda a: cross_entropy(a, labels), random_array(rng, (6, 7), -3.0, 3.0))


def test_cross_entropy_gradient_is_softmax_minus_one_hot():
    rng = np.random.default_rng(6)
    logits_array = random_array(rng, (4, 3), -2.0, 2.0)
    labels = np.array([0, 2, 1, 2])

    logits = Tensor(logits_array)
    cross_entropy(logits, labels).backward()

    softmax = np.exp(logits_array) / np.exp(logits_array).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(logits.grad, (softmax - one_hot(labels, 3)) / 4)


def test_cross_entropy_matches_pytorch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    logits_array = random_array(rng, (32, 10), -5.0, 5.0)
    labels = rng.integers(0, 10, size=32)

    logits = Tensor(logits_array)
    loss = cross_entropy(logits, labels)
    loss.backward()

    torch_logits = torch.tensor(logits_array, requires_grad=True)
    torch_loss = torch.nn.functional.cross_entropy(torch_logits, torch.tensor(labels))
    torch_loss.backward()

    np.testing.assert_allclose(loss.data, torch_loss.item(), rtol=1e-12)
    np.testing.assert_allclose(logits.grad, torch_logits.grad.numpy(), rtol=1e-10, atol=1e-12)


def test_cross_entropy_rejects_float_labels():
    with pytest.raises(TypeError):
        cross_entropy(Tensor(np.zeros((2, 3))), np.array([0.0, 1.0]))


def test_cross_entropy_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        cross_entropy(Tensor(np.zeros((2, 3))), np.array([0, 1, 2]))


# End-to-end sanity check


def test_mlp_can_overfit_a_tiny_batch():
    """If a model cannot memorise 32 samples, the training pipeline is broken."""
    rng = np.random.default_rng(8)
    inputs = Tensor(rng.normal(size=(32, 20)))
    labels = rng.integers(0, 4, size=32)
    hidden_layer = Linear(20, 64, rng=rng)
    output_layer = Linear(64, 4, rng=rng)
    parameters = hidden_layer.parameters() + output_layer.parameters()

    for _ in range(300):
        loss = cross_entropy(output_layer(hidden_layer(inputs).relu()), labels)
        loss.backward()
        sgd_update(parameters, learning_rate=0.5)

    logits = output_layer(hidden_layer(inputs).relu()).data
    assert loss.data < 0.05
    np.testing.assert_array_equal(logits.argmax(axis=1), labels)
