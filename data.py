"""Download and parse the MNIST handwritten digit dataset."""

import gzip
import urllib.request
from pathlib import Path

import numpy as np

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# IDX magic numbers: two zero bytes, a type code (0x08 = unsigned byte), then the rank.
IMAGES_MAGIC = 0x00000803
LABELS_MAGIC = 0x00000801


def download_mnist(data_dir="data"):
    """Download any missing MNIST files into data_dir."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILES.values():
        path = data_dir / filename
        if path.exists():
            continue
        print(f"Downloading {filename} ...")
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(MIRROR + filename, partial)
        partial.replace(path)


def parse_idx(path, expected_magic):
    """Read a gzipped IDX file of unsigned bytes into a NumPy array."""
    with gzip.open(path, "rb") as f:
        raw = f.read()

    magic = int.from_bytes(raw[0:4], "big")
    if magic != expected_magic:
        raise ValueError(f"{path}: magic number {magic:#010x}, expected {expected_magic:#010x}")

    rank = raw[3]
    header_size = 4 + 4 * rank
    shape = tuple(
        int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(rank)
    )
    data = np.frombuffer(raw, dtype=np.uint8, offset=header_size)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: header says {shape}, but found {data.size} values")
    return data.reshape(shape)


def load_split(data_dir, images_name, labels_name):
    images = parse_idx(Path(data_dir) / FILES[images_name], IMAGES_MAGIC)
    labels = parse_idx(Path(data_dir) / FILES[labels_name], LABELS_MAGIC)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"{images.shape[0]} images but {labels.shape[0]} labels")

    # Flatten each image to a vector and scale pixels to [0, 1].
    images = images.reshape(images.shape[0], -1).astype(np.float64) / 255.0
    return images, labels.astype(np.int64)


def load_mnist(data_dir="data", validation_size=5000, seed=0, download=True):
    """Return train, validation, and test splits as (images, labels) pairs.

    The validation set is a fixed random subset of the official training set,
    so hyperparameters can be chosen without looking at the test set.
    """
    if download:
        download_mnist(data_dir)

    train_images, train_labels = load_split(data_dir, "train_images", "train_labels")
    test_images, test_labels = load_split(data_dir, "test_images", "test_labels")

    order = np.random.default_rng(seed).permutation(train_images.shape[0])
    validation_index = order[:validation_size]
    train_index = order[validation_size:]

    return {
        "train": (train_images[train_index], train_labels[train_index]),
        "validation": (train_images[validation_index], train_labels[validation_index]),
        "test": (test_images, test_labels),
    }
