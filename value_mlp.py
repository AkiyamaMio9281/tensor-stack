"""Train the XOR MLP with the scalar autograd engine and check it against mlp.py."""

import time

import numpy as np

from autograd import Value
from mlp import init_params, mlp_forward, mlp_gradients, targets, x


def to_values(params):
    """Wrap every entry of each NumPy parameter in a Value, keeping its shape."""
    return {
        name: np.vectorize(Value, otypes=[object])(array)
        for name, array in params.items()
    }


def grads_of(value_params):
    """Collect the .grad of every Value into float arrays shaped like the parameters."""
    return {
        name: np.vectorize(lambda v: v.grad, otypes=[float])(array)
        for name, array in value_params.items()
    }


def data_of(value_params):
    """Collect the .data of every Value into float arrays shaped like the parameters."""
    return {
        name: np.vectorize(lambda v: v.data, otypes=[float])(array)
        for name, array in value_params.items()
    }


def forward(inputs, vp):
    """Run one sample through the MLP using only Value operations."""
    num_inputs, hidden_size = vp["W1"].shape

    hidden = []
    for j in range(hidden_size):
        pre_activation = vp["b1"][j]
        for i in range(num_inputs):
            pre_activation = pre_activation + float(inputs[i]) * vp["W1"][i, j]
        hidden.append(pre_activation.relu())

    prediction = vp["b2"][0]
    for j in range(hidden_size):
        prediction = prediction + hidden[j] * vp["W2"][j, 0]
    return prediction


def mse_loss(vp):
    """Mean squared error over the four XOR samples, built as one Value graph."""
    total = Value(0.0)
    for inputs, target in zip(x, targets[:, 0]):
        error = forward(inputs, vp) - float(target)
        total = total + error * error
    return total * (1.0 / len(x))


def parameters(vp):
    """Flatten all Value parameters into one list."""
    return [value for array in vp.values() for value in array.flat]


if __name__ == "__main__":
    numpy_params = init_params()
    vp = to_values(numpy_params)

    # Gradients on the first step must match the hand-derived backpropagation.
    predictions, hidden, hidden_pre_activation = mlp_forward(x, numpy_params)
    expected_grads = mlp_gradients(
        x, targets, predictions, hidden, hidden_pre_activation, numpy_params
    )
    expected_loss = np.mean((predictions - targets) ** 2)

    loss = mse_loss(vp)
    loss.backward()

    np.testing.assert_allclose(loss.data, expected_loss, rtol=1e-12)
    autograd_grads = grads_of(vp)
    for name in expected_grads:
        np.testing.assert_allclose(
            autograd_grads[name],
            expected_grads[name],
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Autograd gradient mismatch for {name}",
        )
    print("Autograd gradients match mlp_gradients on the first step.")

    learning_rate = 0.1
    num_steps = 2000
    start = time.perf_counter()

    for step in range(num_steps):
        loss = mse_loss(vp)
        loss.backward()

        if step in (0, 100, 500):
            print(f"Step {step}: loss = {loss.data:.6f}")

        for p in parameters(vp):
            p.data -= learning_rate * p.grad

    elapsed = time.perf_counter() - start

    final_params = data_of(vp)
    final_predictions, _, _ = mlp_forward(x, final_params)
    final_loss = np.mean((final_predictions - targets) ** 2)
    classes = (final_predictions >= 0.5).astype(int)

    print(f"Final loss: {final_loss:.3e}")
    print("Predicted classes:", classes.ravel())
    print(f"Training took {elapsed:.1f}s")

    np.testing.assert_array_equal(classes, targets.astype(int))
    assert final_loss < 1e-4
