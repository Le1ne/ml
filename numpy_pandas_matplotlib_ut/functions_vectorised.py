import numpy as np
from typing import Tuple


def sum_non_neg_diag(X: np.ndarray) -> int:
    return -1 if not np.any(np.diag(X) >= 0) else np.sum(np.diag(X)[np.diag(X) >= 0])


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    return np.array_equal(np.sort(x), np.sort(y))


def max_prod_mod_3(x: np.ndarray) -> int:
    neighbours = np.array(x[1:] * x[:-1])
    filtered = neighbours[(x[1:] % 3 == 0) | (x[:-1] % 3 == 0)]

    return -1 if filtered.size == 0 else filtered.max()


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(image * weights, axis=2)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    x_processed = np.repeat(x[:, 0], x[:, 1])
    y_processed = np.repeat(y[:, 0], y[:, 1])

    return -1 if x_processed.size != y_processed.size else np.dot(x_processed, y_processed)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    scalar_prod = np.dot(X, Y.T)
    x_norms = np.linalg.norm(X, axis=1, keepdims=True)
    y_norms = np.linalg.norm(Y, axis=1)
    norms_prod = np.outer(x_norms, y_norms)

    with np.errstate(divide='ignore', invalid='ignore'):
        distances = scalar_prod / norms_prod

    distances[~np.isfinite(distances)] = 1

    return distances
