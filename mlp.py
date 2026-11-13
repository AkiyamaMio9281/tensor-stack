import numpy as np


def mlp_forward(x, params):
    hidden_pre_activation = x @ params["W1"] + params["b1"]
    hidden = np.maximum(0.0, hidden_pre_activation)
    return hidden @ params["W2"] + params["b2"]


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

predictions = mlp_forward(x, params)
loss = np.mean((predictions - targets) ** 2)

print("Predictions:", predictions.ravel())
print("Loss:", loss)

np.testing.assert_allclose(predictions, targets)