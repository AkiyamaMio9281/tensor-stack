import numpy as np


def mlp_forward(x, params):
    hidden_pre_activation = x @ params["W1"] + params["b1"]
    hidden = np.maximum(0.0, hidden_pre_activation)
    predictions = hidden @ params["W2"] + params["b2"]
    return predictions, hidden, hidden_pre_activation

def mlp_gradients(x, targets, predictions, hidden, hidden_pre_activation, params):
    errors = predictions - targets
    d_predictions = 2.0 * errors / errors.size

    # Compute output-layer gradients.
    dW2 = hidden.T @ d_predictions
    db2 = np.sum(d_predictions, axis=0)

    # Propagate gradients through the output weights and ReLU.
    d_hidden = d_predictions @ params["W2"].T
    d_hidden_pre_activation = d_hidden * (hidden_pre_activation > 0.0)

    # Compute first-layer gradients.
    dW1 = x.T @ d_hidden_pre_activation
    db1 = np.sum(d_hidden_pre_activation, axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def mlp_loss(x, targets, params):
    predictions, _, _ = mlp_forward(x, params)
    return np.mean((predictions - targets) ** 2)


def numerical_gradients(x, targets, params, epsilon=1e-5):
    grads = {}

    for name, values in params.items():
        grad = np.zeros_like(values)

        for index in np.ndindex(values.shape):
            perturbation = np.zeros_like(values)
            perturbation[index] = epsilon

            plus_params = {**params, name: values + perturbation}
            minus_params = {**params, name: values - perturbation}

            loss_plus = mlp_loss(x, targets, plus_params)
            loss_minus = mlp_loss(x, targets, minus_params)
            grad[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

        grads[name] = grad

    return grads

x = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

targets = np.array([[0.0], [1.0], [1.0], [0.0]])

# Use known weights to verify the forward pass.
params = {
    "W1": np.array([[1.0, 1.0], [1.0, 1.0]]),
    "b1": np.array([0.0, -1.0]),
    "W2": np.array([[1.0], [-2.0]]),
    "b2": np.array([0.0]),
}

predictions, _, _ = mlp_forward(x, params)
loss = np.mean((predictions - targets) ** 2)

print("Predictions:", predictions.ravel())
print("Loss:", loss)

np.testing.assert_allclose(predictions, targets)

gradient_test_params = {**params, "b2": np.array([0.25])}
predictions, hidden, hidden_pre_activation = mlp_forward(x, gradient_test_params)
grads = mlp_gradients(
    x, targets, predictions, hidden, hidden_pre_activation, gradient_test_params
)

np.testing.assert_allclose(grads["W2"], [[0.5], [0.125]])
np.testing.assert_allclose(grads["b2"], [0.5])
np.testing.assert_allclose(grads["W1"], [[0.25, -0.25], [0.25, -0.25]])
np.testing.assert_allclose(grads["b1"], [0.375, -0.25])

print("MLP gradient checks passed.")

check_params = {
    "W1": np.array([[0.5, -0.4], [0.3, 0.8]]),
    "b1": np.array([-0.2, 0.1]),
    "W2": np.array([[0.7], [-1.2]]),
    "b2": np.array([0.25]),
}

predictions, hidden, hidden_pre_activation = mlp_forward(x, check_params)
analytical_grads = mlp_gradients(
    x, targets, predictions, hidden, hidden_pre_activation, check_params
)
numerical_grads = numerical_gradients(x, targets, check_params)

for name in check_params:
    np.testing.assert_allclose(
        analytical_grads[name],
        numerical_grads[name],
        rtol=1e-5,
        atol=1e-8,
        err_msg=f"Gradient check failed for {name}",
    )

print("Finite-difference gradient check passed.")
