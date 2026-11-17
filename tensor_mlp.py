"""Train the XOR MLP with Tensor autograd and check it step by step against mlp.py."""

import numpy as np

from mlp import init_params, mlp_forward, mlp_gradients, targets, x
from nn import Linear
from optim import sgd_step, sgd_update
from tensor import Tensor


class MLP:
    def __init__(self, params):
        self.hidden_layer = Linear.from_arrays(params["W1"], params["b1"])
        self.output_layer = Linear.from_arrays(params["W2"], params["b2"])

    def __call__(self, inputs):
        return self.output_layer(self.hidden_layer(inputs).relu())

    def parameters(self):
        return self.hidden_layer.parameters() + self.output_layer.parameters()

    def named_parameters(self):
        names = ["W1", "b1", "W2", "b2"]
        return dict(zip(names, self.parameters()))


def mse_loss(predictions, targets):
    return ((predictions - targets) ** 2).mean()


if __name__ == "__main__":
    numpy_params = init_params()
    model = MLP(numpy_params)
    inputs = Tensor(x)

    learning_rate = 0.1
    num_steps = 2000
    max_param_gap = 0.0

    # Train both implementations in lockstep and compare after every step.
    for step in range(num_steps):
        predictions, hidden, hidden_pre_activation = mlp_forward(x, numpy_params)
        numpy_grads = mlp_gradients(
            x, targets, predictions, hidden, hidden_pre_activation, numpy_params
        )

        loss = mse_loss(model(inputs), targets)
        loss.backward()

        if step == 0:
            for name, parameter in model.named_parameters().items():
                np.testing.assert_allclose(
                    parameter.grad,
                    numpy_grads[name],
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"Tensor gradient mismatch for {name}",
                )
            print("Tensor gradients match mlp_gradients on the first step.")

        if step in (0, 100, 500):
            print(f"Step {step}: loss = {loss.data:.6f}")

        sgd_step(numpy_params, numpy_grads, learning_rate)
        sgd_update(model.parameters(), learning_rate)

        for name, parameter in model.named_parameters().items():
            gap = np.max(np.abs(parameter.data - numpy_params[name]))
            max_param_gap = max(max_param_gap, gap)

    final_predictions = model(inputs).data
    final_loss = np.mean((final_predictions - targets) ** 2)
    classes = (final_predictions >= 0.5).astype(int)

    print(f"Final loss: {final_loss:.3e}")
    print("Predicted classes:", classes.ravel())
    print(f"Largest parameter gap vs NumPy training: {max_param_gap:.1e}")

    np.testing.assert_array_equal(classes, targets.astype(int))
    assert final_loss < 1e-4
    assert max_param_gap < 1e-10
