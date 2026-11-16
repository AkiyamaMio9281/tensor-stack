class Value:
    def __init__(self, data, parents=()):
        self.data = float(data)
        self.grad = 0.0
        self._parents = parents
        self._backward = lambda: None

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other))

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other))

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

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