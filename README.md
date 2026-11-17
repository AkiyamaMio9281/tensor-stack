# tensor-stack

Practicing machine learning algorithms by implementing them from scratch.

So far:

- Linear regression in plain NumPy, with hand-derived gradients and gradient descent.
- A two-layer MLP that learns XOR, with hand-derived backpropagation checked by finite differences.
- A scalar reverse-mode autograd engine (`autograd.py`) that trains the same MLP.
- A NumPy-array autograd engine (`tensor.py`) with broadcasting-aware gradients, a `Linear` layer (`nn.py`), and SGD. It reproduces the hand-derived XOR training exactly, and its gradients are tested against finite differences and PyTorch.

## Run

```bash
pip install -r requirements.txt
python train.py        # linear regression
python mlp.py          # XOR MLP with hand-derived gradients
python value_mlp.py    # XOR MLP with scalar autograd
python tensor_mlp.py   # XOR MLP with Tensor autograd
```

## Test

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Install `torch` as well to run the PyTorch comparison tests; they are skipped otherwise.
