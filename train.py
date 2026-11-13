import numpy as np

# Create five training examples.
x = np.array([-2, -1, 0, 1, 2], dtype=float)
y = 3 * x + 2

# Start with a weight and bias of zero.
w = 0.0
b = 0.0

# Make predictions and compute the mean squared error.
y_pred = w * x + b
errors = y_pred - y
loss = np.mean(errors ** 2)

print("Actual values:", y)
print("Predictions:", y_pred)
print("Loss:", loss)

# Compute the gradients of the loss with respect to w and b.
# dL/dw = 2 * mean(error * x), dL/db = 2 * mean(error)
dw = 2 * np.mean(errors * x)
db = 2 * np.mean(errors)

print("Weight gradient:", dw)
print("Bias gradient:", db)

learning_rate = 0.1

w -= learning_rate * dw
b -= learning_rate * db

# Recompute predictions and loss with the updated parameters.
y_pred = w * x + b
errors = y_pred - y
new_loss = np.mean(errors ** 2)

print(f"Updated weight: {w:.2f}")
print(f"Updated bias: {b:.2f}")
print(f"Loss before update: {loss:.2f}")
print(f"Loss after update: {new_loss:.2f}")