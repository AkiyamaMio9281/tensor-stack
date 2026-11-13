import numpy as np


def mlp_forward(x, params):
    hidden_pre_activation = x @ params["W1"] + params["b1"]
    hidden = np.maximum(0.0, hidden_pre_activation)
    predictions = hidden @ params["W2"] + params["b2"]
    return predictions, hidden

def output_layer_gradients(predictions, targets, hidden):
    errors = predictions - targets
    d_predictions = 2.0 * errors / errors.size

    dW2 = hidden.T @ d_predictions
    db2 = np.sum(d_predictions, axis=0)
    return dW2, db2

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

predictions, _ = mlp_forward(x, params)
loss = np.mean((predictions - targets) ** 2)

print("Predictions:", predictions.ravel())
print("Loss:", loss)

np.testing.assert_allclose(predictions, targets)

gradient_test_params = {**params, "b2": np.array([0.25])}
predictions, hidden = mlp_forward(x, gradient_test_params)
dW2, db2 = output_layer_gradients(predictions, targets, hidden)

np.testing.assert_allclose(dW2, [[0.5], [0.125]])
np.testing.assert_allclose(db2, [0.5])

print("Output layer gradient check passed.")