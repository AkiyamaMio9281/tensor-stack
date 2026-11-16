import numpy as np
from optim import sgd_step


def predict(x, params):
    return params["w"] * x + params["b"]


def mse_loss(errors):
    return np.mean(errors ** 2)


def linear_gradients(x, errors):
    return {
        "w": 2 * np.mean(errors * x),
        "b": 2 * np.mean(errors),
    }



x = np.array([-2, -1, 0, 1, 2], dtype=float)
y = 3 * x + 2

params = {"w": 0.0, "b": 0.0}
learning_rate = 0.1
num_steps = 50

for step in range(num_steps):
    predictions = predict(x, params)
    errors = predictions - y
    loss = mse_loss(errors)

    if step in (0, 1):
        print(f"Loss after {step} updates: {loss:.2f}")

    grads = linear_gradients(x, errors)
    sgd_step(params, grads, learning_rate)

final_loss = mse_loss(predict(x, params) - y)

print(f"Final weight: {params['w']:.6f}")
print(f"Final bias: {params['b']:.6f}")
print(f"Final loss: {final_loss:.2e}")

