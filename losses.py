"""Loss functions built on Tensor."""

import numpy as np


def one_hot(labels, num_classes):
    encoded = np.zeros((labels.shape[0], num_classes))
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def cross_entropy(logits, labels):
    """Mean cross-entropy between (N, C) logits and N integer class labels."""
    labels = np.asarray(labels)
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels must be integer class indices")
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("expected logits of shape (N, C) and labels of shape (N,)")

    targets = one_hot(labels, logits.shape[1])
    return -(logits.log_softmax(axis=1) * targets).sum(axis=1).mean()
