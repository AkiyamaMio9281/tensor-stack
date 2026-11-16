def sgd_step(params, grads, learning_rate):
    for name in params:
        params[name] -= learning_rate * grads[name]