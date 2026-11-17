import numpy as np
import pytest

from gradcheck import away_from_zero, check_gradients, random_array
from tensor import Tensor, unbroadcast


# unbroadcast


@pytest.mark.parametrize(
    "grad_shape, target_shape",
    [
        ((4, 8), (4, 8)),
        ((4, 8), (8,)),
        ((4, 8), (1, 8)),
        ((4, 8), (4, 1)),
        ((4, 8), ()),
        ((2, 3, 5), (3, 1)),
    ],
)
def test_unbroadcast_matches_manual_sum(grad_shape, target_shape):
    grad = np.arange(np.prod(grad_shape), dtype=float).reshape(grad_shape)
    reduced = unbroadcast(grad, target_shape)

    assert np.shape(reduced) == target_shape
    # Every gradient entry must land somewhere: totals are preserved.
    np.testing.assert_allclose(np.sum(reduced), grad.sum())
    # Broadcasting the reduced shape back must reproduce the original shape.
    assert np.broadcast_shapes(target_shape, grad_shape) == grad_shape


def test_unbroadcast_bias_gradient_sums_rows():
    grad = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_allclose(unbroadcast(grad, (2,)), [9.0, 12.0])
    np.testing.assert_allclose(unbroadcast(grad, (3, 1)), [[3.0], [7.0], [11.0]])


# Elementwise operations


BROADCAST_SHAPES = [
    ((4, 8), (4, 8)),
    ((4, 8), (8,)),
    ((4, 8), (1, 8)),
    ((4, 8), (4, 1)),
    ((4, 8), ()),
    ((3, 1), (1, 5)),
    ((8,), (4, 8)),
]


@pytest.mark.parametrize("shape_a, shape_b", BROADCAST_SHAPES)
@pytest.mark.parametrize(
    "op",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
    ],
    ids=["add", "sub", "mul", "div"],
)
def test_binary_ops_with_broadcasting(op, shape_a, shape_b):
    rng = np.random.default_rng(1)
    check_gradients(op, away_from_zero(rng, shape_a), away_from_zero(rng, shape_b))


@pytest.mark.parametrize(
    "op",
    [
        lambda a: -a,
        lambda a: a ** 2,
        lambda a: a ** 3,
        lambda a: a ** -1,
        lambda a: a ** 0.5,
        lambda a: a.relu(),
        lambda a: 2.0 + a,
        lambda a: 2.0 - a,
        lambda a: 2.0 * a,
        lambda a: 2.0 / a,
        lambda a: a / 4.0,
        lambda a: a * a + a,
    ],
    ids=[
        "neg", "square", "cube", "reciprocal", "sqrt", "relu",
        "radd", "rsub", "rmul", "rtruediv", "div_scalar", "reuse",
    ],
)
def test_unary_ops(op):
    rng = np.random.default_rng(2)
    # Positive inputs keep sqrt and reciprocal defined; ReLU on mixed signs is tested below.
    check_gradients(op, np.abs(away_from_zero(rng, (3, 4))))


def test_relu_on_mixed_signs():
    rng = np.random.default_rng(3)
    check_gradients(lambda a: a.relu(), away_from_zero(rng, (5, 6)))


def test_numpy_array_on_the_left_returns_tensor():
    rng = np.random.default_rng(4)
    array = random_array(rng, (2, 3))
    tensor = Tensor(random_array(rng, (2, 3)))

    for result in (array + tensor, array - tensor, array * tensor, array / tensor):
        assert isinstance(result, Tensor)


# Matrix multiplication


@pytest.mark.parametrize("shape_a, shape_b", [((4, 3), (3, 5)), ((1, 7), (7, 1)), ((6, 2), (2, 2))])
def test_matmul(shape_a, shape_b):
    rng = np.random.default_rng(5)
    check_gradients(lambda a, b: a @ b, random_array(rng, shape_a), random_array(rng, shape_b))


def test_matmul_rejects_non_2d():
    with pytest.raises(ValueError):
        Tensor(np.ones(3)) @ Tensor(np.ones((3, 2)))


def test_linear_layer_expression():
    rng = np.random.default_rng(6)
    check_gradients(
        lambda x, w, b: (x @ w + b).relu(),
        random_array(rng, (4, 3)),
        random_array(rng, (3, 5)),
        random_array(rng, (5,)),
    )


# Reductions


@pytest.mark.parametrize(
    "op",
    [
        lambda a: a.sum(),
        lambda a: a.sum(axis=0),
        lambda a: a.sum(axis=1),
        lambda a: a.sum(axis=-1),
        lambda a: a.sum(axis=1, keepdims=True),
        lambda a: a.sum(axis=(0, 2)),
        lambda a: a.mean(),
        lambda a: a.mean(axis=0),
        lambda a: a.mean(axis=-1, keepdims=True),
        lambda a: a.mean(axis=(1, 2)),
    ],
    ids=[
        "sum", "sum0", "sum1", "sum-1", "sum1_keep", "sum02",
        "mean", "mean0", "mean-1_keep", "mean12",
    ],
)
def test_reductions(op):
    rng = np.random.default_rng(7)
    check_gradients(op, random_array(rng, (2, 3, 4)))


def test_mean_matches_numpy():
    rng = np.random.default_rng(8)
    array = random_array(rng, (3, 4))
    np.testing.assert_allclose(Tensor(array).mean(axis=0).data, array.mean(axis=0))
    np.testing.assert_allclose(Tensor(array).mean().data, array.mean())


# Graph behaviour


def test_node_used_twice_accumulates_gradient():
    a = Tensor(np.array([1.0, -2.0, 3.0]))
    (a * a + a).sum().backward()
    np.testing.assert_allclose(a.grad, 2.0 * a.data + 1.0)


def test_deep_graph_does_not_overflow_the_stack():
    x = Tensor(np.array(1.0))
    y = x
    for _ in range(10_000):
        y = y + x
    y.backward()
    assert x.grad == 10_001.0


def test_backward_twice_gives_the_same_gradient():
    a = Tensor(np.array([0.5, -1.5]))
    loss = (a * a).sum()
    loss.backward()
    first = a.grad.copy()
    loss.backward()
    np.testing.assert_allclose(a.grad, first)


def test_backward_requires_single_element():
    with pytest.raises(ValueError):
        Tensor(np.ones(3)).backward()


def test_pow_rejects_tensor_exponent():
    with pytest.raises(TypeError):
        Tensor(np.ones(2)) ** Tensor(np.ones(2))


def test_leaf_does_not_alias_input_array():
    array = np.ones(3)
    tensor = Tensor(array)
    tensor.data -= 1.0
    np.testing.assert_allclose(array, np.ones(3))


# Comparison with PyTorch


def test_two_layer_mlp_matches_pytorch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(9)
    x = random_array(rng, (16, 5))
    targets = random_array(rng, (16, 1))
    arrays = {
        "W1": random_array(rng, (5, 8)),
        "b1": random_array(rng, (8,)),
        "W2": random_array(rng, (8, 1)),
        "b2": random_array(rng, (1,)),
    }

    params = {name: Tensor(a) for name, a in arrays.items()}
    hidden = (Tensor(x) @ params["W1"] + params["b1"]).relu()
    predictions = hidden @ params["W2"] + params["b2"]
    loss = ((predictions - targets) ** 2).mean()
    loss.backward()

    torch_params = {
        name: torch.tensor(a, dtype=torch.float64, requires_grad=True)
        for name, a in arrays.items()
    }
    torch_x = torch.tensor(x, dtype=torch.float64)
    torch_targets = torch.tensor(targets, dtype=torch.float64)
    torch_hidden = torch.relu(torch_x @ torch_params["W1"] + torch_params["b1"])
    torch_predictions = torch_hidden @ torch_params["W2"] + torch_params["b2"]
    torch_loss = ((torch_predictions - torch_targets) ** 2).mean()
    torch_loss.backward()

    np.testing.assert_allclose(loss.data, torch_loss.item(), rtol=1e-12)
    for name in arrays:
        np.testing.assert_allclose(
            params[name].grad,
            torch_params[name].grad.numpy(),
            rtol=1e-10,
            atol=1e-12,
            err_msg=name,
        )


def test_reductions_and_division_match_pytorch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(10)
    a_array = away_from_zero(rng, (3, 4))
    b_array = away_from_zero(rng, (4,))

    a, b = Tensor(a_array), Tensor(b_array)
    loss = ((a / b).sum(axis=1) ** 2).mean() + (a * b).mean(axis=0).sum()
    loss.backward()

    ta = torch.tensor(a_array, requires_grad=True)
    tb = torch.tensor(b_array, requires_grad=True)
    torch_loss = ((ta / tb).sum(dim=1) ** 2).mean() + (ta * tb).mean(dim=0).sum()
    torch_loss.backward()

    np.testing.assert_allclose(loss.data, torch_loss.item(), rtol=1e-12)
    np.testing.assert_allclose(a.grad, ta.grad.numpy(), rtol=1e-10)
    np.testing.assert_allclose(b.grad, tb.grad.numpy(), rtol=1e-10)
