import gzip

import numpy as np
import pytest

from data import FILES, IMAGES_MAGIC, LABELS_MAGIC, load_mnist, parse_idx


def write_idx(path, array, magic):
    rank = magic & 0xFF
    assert array.ndim == rank
    header = magic.to_bytes(4, "big") + b"".join(
        size.to_bytes(4, "big") for size in array.shape
    )
    with gzip.open(path, "wb") as f:
        f.write(header + array.astype(np.uint8).tobytes())


def write_fake_mnist(data_dir, num_train, num_test, rng):
    for split, count in (("train", num_train), ("test", num_test)):
        images = rng.integers(0, 256, size=(count, 2, 3))
        labels = rng.integers(0, 10, size=count)
        write_idx(data_dir / FILES[f"{split}_images"], images, IMAGES_MAGIC)
        write_idx(data_dir / FILES[f"{split}_labels"], labels, LABELS_MAGIC)


def test_parse_idx_round_trip(tmp_path):
    images = np.arange(24, dtype=np.uint8).reshape(4, 2, 3)
    path = tmp_path / "images.gz"
    write_idx(path, images, IMAGES_MAGIC)

    np.testing.assert_array_equal(parse_idx(path, IMAGES_MAGIC), images)


def test_parse_idx_rejects_wrong_magic(tmp_path):
    path = tmp_path / "labels.gz"
    write_idx(path, np.arange(5), LABELS_MAGIC)

    with pytest.raises(ValueError, match="magic"):
        parse_idx(path, IMAGES_MAGIC)


def test_parse_idx_rejects_truncated_data(tmp_path):
    path = tmp_path / "labels.gz"
    header = LABELS_MAGIC.to_bytes(4, "big") + (10).to_bytes(4, "big")
    with gzip.open(path, "wb") as f:
        f.write(header + bytes(7))

    with pytest.raises(ValueError, match="header"):
        parse_idx(path, LABELS_MAGIC)


def test_load_mnist_splits_and_scales(tmp_path):
    write_fake_mnist(tmp_path, num_train=20, num_test=6, rng=np.random.default_rng(0))
    splits = load_mnist(tmp_path, validation_size=5, download=False)

    train_images, train_labels = splits["train"]
    validation_images, validation_labels = splits["validation"]
    test_images, test_labels = splits["test"]

    assert train_images.shape == (15, 6)
    assert validation_images.shape == (5, 6)
    assert test_images.shape == (6, 6)
    assert train_labels.dtype == np.int64

    for images in (train_images, validation_images, test_images):
        assert images.min() >= 0.0 and images.max() <= 1.0


def test_validation_split_is_disjoint_and_reproducible(tmp_path):
    write_fake_mnist(tmp_path, num_train=50, num_test=5, rng=np.random.default_rng(1))
    # Tag each training image by its index so rows can be traced back.
    images = np.zeros((50, 2, 3), dtype=np.uint8)
    images[:, 0, 0] = np.arange(50)
    write_idx(tmp_path / FILES["train_images"], images, IMAGES_MAGIC)

    first = load_mnist(tmp_path, validation_size=10, seed=3, download=False)
    second = load_mnist(tmp_path, validation_size=10, seed=3, download=False)

    train_ids = set(np.round(first["train"][0][:, 0] * 255).astype(int))
    validation_ids = set(np.round(first["validation"][0][:, 0] * 255).astype(int))
    assert train_ids.isdisjoint(validation_ids)
    assert len(train_ids | validation_ids) == 50
    np.testing.assert_array_equal(first["validation"][0], second["validation"][0])


def test_load_mnist_rejects_mismatched_counts(tmp_path):
    write_fake_mnist(tmp_path, num_train=10, num_test=4, rng=np.random.default_rng(2))
    write_idx(tmp_path / FILES["test_labels"], np.arange(3), LABELS_MAGIC)

    with pytest.raises(ValueError, match="labels"):
        load_mnist(tmp_path, validation_size=2, download=False)
