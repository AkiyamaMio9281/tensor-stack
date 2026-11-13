import numpy as np

x = np.array([-2, -1, 0, 1, 2], dtype=float)
y = 3 * x + 2

w = 0.0
b = 0.0
learning_rate = 0.1
num_steps = 50

for step in range(num_steps):
    # Compute predictions and loss with the current parameters.
    y_pred = w * x + b
    errors = y_pred - y
    loss = np.mean(errors ** 2)

    if step in (0, 1):
        print(f"Loss after {step} updates: {loss:.2f}")

    # Compute both gradients before updating either parameter.
    dw = 2 * np.mean(errors * x)
    db = 2 * np.mean(errors)

    w -= learning_rate * dw
    b -= learning_rate * db

# Evaluate the parameters after the last update.
final_predictions = w * x + b
final_loss = np.mean((final_predictions - y) ** 2)

print(f"Final weight: {w:.6f}")
print(f"Final bias: {b:.6f}")
print(f"Final loss: {final_loss:.2e}")