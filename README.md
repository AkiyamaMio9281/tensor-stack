# tensor-stack

Practicing machine learning algorithms by implementing them from scratch.

So far:

- Linear regression in plain NumPy, with hand-derived gradients and gradient descent.
- A two-layer MLP that learns XOR, with hand-derived backpropagation checked by finite differences.
- A scalar reverse-mode autograd engine (`autograd.py`) that trains the same MLP.
- A NumPy-array autograd engine (`tensor.py`) with broadcasting-aware gradients, a `Linear` layer (`nn.py`), and SGD. It reproduces the hand-derived XOR training exactly, and its gradients are tested against finite differences and PyTorch.
- Numerically stable `log_softmax` and cross-entropy (`losses.py`), an MNIST loader (`data.py`), and an MLP trained on MNIST (`train_mnist.py`).

## MNIST result

A 784-128-10 MLP with ReLU, trained with plain SGD on the autograd engine above:

| Test accuracy | Validation accuracy | Epochs | Batch size | Learning rate | Training time (CPU) |
|---|---|---|---|---|---|
| 98.04% | 97.60% | 20 | 128 | 0.5 | 26 s |

The learning rate and epoch count were chosen on a 5,000-image validation split held out from the training set; the test set was evaluated once at the end. Per-epoch metrics are in `results/mnist_mlp.json`.

## Run

```bash
pip install -r requirements.txt
python train.py        # linear regression
python mlp.py          # XOR MLP with hand-derived gradients
python value_mlp.py    # XOR MLP with scalar autograd
python tensor_mlp.py   # XOR MLP with Tensor autograd
python train_mnist.py  # MNIST MLP; downloads the dataset into data/ on first run
```

## Test

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Install `torch` as well to run the PyTorch comparison tests; they are skipped otherwise.
