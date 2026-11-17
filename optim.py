def sgd_step(params, grads, learning_rate):
    for name in params:
        params[name] -= learning_rate * grads[name]


def sgd_update(parameters, learning_rate):
    """Apply one SGD step to Tensor parameters using their accumulated .grad."""
    for parameter in parameters:
        parameter.data -= learning_rate * parameter.grad
