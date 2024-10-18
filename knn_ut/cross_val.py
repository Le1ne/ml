import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    fold_size = num_objects // num_folds
    remainder = num_objects % num_folds
    indices = np.arange(num_objects)

    folds = []
    start_idx = 0
    for i in range(num_folds):
        end_idx = start_idx + fold_size + (remainder if i == num_folds - 1 else 0)
        fold = indices[start_idx:end_idx]
        folds.append(fold)
        start_idx = end_idx

    split_pairs = []
    for i in range(num_folds):
        val_indices = folds[i]
        train_indices = np.concatenate([folds[j] for j in range(num_folds) if j != i])
        split_pairs.append((train_indices, val_indices))

    return split_pairs


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    scores = {}

    n_neighbors_params = parameters.get('n_neighbors')
    metrics_params = parameters.get('metrics')
    weights_params = parameters.get('weights')
    normalizers = parameters.get('normalizers')

    for normalizer, normalizer_name in normalizers:
        for n_neighbors in n_neighbors_params:
            for metric in metrics_params:
                for weight in weights_params:
                    knn = knn_class(n_neighbors=n_neighbors, metric=metric, weights=weight)
                    fold_scores = []

                    for train_idx, val_idx in folds:
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]

                        if normalizer:
                            normalizer.fit(X_train)
                            X_train = normalizer.transform(X_train)
                            X_val = normalizer.transform(X_val)

                        knn.fit(X_train, y_train)
                        y_pred = knn.predict(X_val)

                        fold_scores.append(score_function(y_val, y_pred))

                    mean_score = np.mean(fold_scores)
                    scores[(normalizer_name, n_neighbors, metric, weight)] = mean_score

    return scores
