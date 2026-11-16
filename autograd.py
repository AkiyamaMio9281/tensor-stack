class Value:
    def __init__(self, data, parents=()):
        self.data = float(data)
        self.grad = 0.0
        self._parents = parents
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def relu(self):
        active = self.data > 0.0
        out = Value(max(0.0, self.data), (self,))

        def backward():
            if active:
                self.grad += out.grad

        out._backward = backward
        return out

    def backward(self):
        order = []
        visited = set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for parent in node._parents:
                visit(parent)
            order.append(node)

        visit(self)

        # Start a fresh backward pass.
        for node in order:
            node.grad = 0.0
        self.grad = 1.0

        for node in reversed(order):
            node._backward()


if __name__ == "__main__":
    x = Value(3.0)
    u = x * x
    y = u + u * x
    y.backward()

    assert y.data == 36.0
    assert u.grad == 4.0
    assert x.grad == 33.0
    print(f"y = {y.data}, dy/du = {u.grad}, dy/dx = {x.grad}")

    def check_neuron(weight, expected_loss, expected_dw, expected_db):
        w = Value(weight)
        b = Value(0.5)

        prediction = (2.0 * w + b).relu()
        error = prediction - 2.0
        loss = error * error
        loss.backward()

        assert loss.data == expected_loss
        assert w.grad == expected_dw
        assert b.grad == expected_db

    check_neuron(0.5, 0.25, -2.0, -1.0)
    check_neuron(-0.5, 4.0, 0.0, 0.0)
    print("Scalar neuron gradient checks passed.")