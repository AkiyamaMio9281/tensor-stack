"""Train a two-layer MLP on MNIST using the Tensor autograd engine."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data import load_mnist
from losses import cross_entropy
from nn import Linear
from optim import sgd_update
from tensor import Tensor


class MLP:
    def __init__(self, in_features, hidden_features, out_features, rng):
        self.hidden_layer = Linear(in_features, hidden_features, rng=rng)
        self.output_layer = Linear(hidden_features, out_features, rng=rng)

    def __call__(self, inputs):
        return self.output_layer(self.hidden_layer(inputs).relu())

    def parameters(self):
        return self.hidden_layer.parameters() + self.output_layer.parameters()


def accuracy(model, images, labels, batch_size=1000):
    """Fraction of images whose highest-scoring class equals the label."""
    correct = 0
    for start in range(0, images.shape[0], batch_size):
        logits = model(Tensor(images[start : start + batch_size])).data
        correct += int((logits.argmax(axis=1) == labels[start : start + batch_size]).sum())
    return correct / images.shape[0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults were chosen by validation accuracy over learning rates 0.05 to 0.5.
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--metrics", default="results/mnist_mlp.json")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    splits = load_mnist(args.data_dir, seed=args.seed)
    train_images, train_labels = splits["train"]
    validation_images, validation_labels = splits["validation"]
    test_images, test_labels = splits["test"]

    model = MLP(train_images.shape[1], args.hidden, 10, rng)
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        order = rng.permutation(train_images.shape[0])
        total_loss = 0.0

        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            loss = cross_entropy(model(Tensor(train_images[batch])), train_labels[batch])
            loss.backward()
            sgd_update(model.parameters(), args.learning_rate)
            total_loss += float(loss.data) * len(batch)

        train_loss = total_loss / len(order)
        validation_accuracy = accuracy(model, validation_images, validation_labels)
        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_accuracy": validation_accuracy,
                "seconds": epoch_seconds,
            }
        )
        print(
            f"Epoch {epoch:2d}: train loss {train_loss:.4f}, "
            f"validation accuracy {validation_accuracy:.2%} ({epoch_seconds:.1f}s)"
        )

    # The test set is evaluated once, after all training decisions are made.
    test_accuracy = accuracy(model, test_images, test_labels)
    total_seconds = time.perf_counter() - start_time
    print(f"Test accuracy: {test_accuracy:.2%} (training took {total_seconds:.0f}s)")

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "config": vars(args),
        "history": history,
        "test_accuracy": test_accuracy,
        "total_seconds": total_seconds,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
