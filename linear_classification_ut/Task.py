import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype
        self.categories_ = {}

    def fit(self, X, Y=None):
        for column in X.columns:
            self.categories_[column] = sorted(X[column].unique())

    def transform(self, X):
        n_objects = X.shape[0]
        n_features = sum(len(categories) for categories in self.categories_.values())
        result = np.zeros((n_objects, n_features), dtype=self.dtype)

        feature_start_idx = 0
        for column, categories in self.categories_.items():
            for i, category in enumerate(categories):
                indices = np.where(X[column] == category)
                result[indices, feature_start_idx + i] = 1
            feature_start_idx += len(categories)

        return result

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.counters = {}

    def fit(self, X, Y):
        for column in X.columns:
            unique_values = X[column].unique()
            self.counters[column] = {}
            for value in unique_values:
                indices = X[column] == value
                self.counters[column][value] = [Y[indices].mean(), np.mean(indices)]

    def transform(self, X, a=1e-5, b=1e-5):
        n_objects, n_features = X.shape
        result = np.zeros((n_objects, 3 * n_features), dtype=self.dtype)

        for i, column in enumerate(X.columns):
            for j in range(n_objects):
                value = X.iloc[j, i]
                mean_target, fraction = self.counters[column][value]
                result[j, 3 * i] = mean_target
                result[j, 3 * i + 1] = fraction
                result[j, 3 * i + 2] = (mean_target + a) / (fraction + b)

        return result

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        self.fold_counters = []

    def fit(self, X, Y, seed=1):
        for fold_idx, rest_idx in group_k_fold(X.shape[0], self.n_folds, seed):
            fold_counter = {}
            X_fold, Y_fold = X.iloc[rest_idx], Y.iloc[rest_idx]
            for column in X.columns:
                unique_values = X_fold[column].unique()
                fold_counter[column] = {}
                for value in unique_values:
                    indices = X_fold[column] == value
                    fold_counter[column][value] = [Y_fold[indices].mean(), np.mean(indices)]
            self.fold_counters.append((fold_idx, fold_counter))

    def transform(self, X, a=1e-5, b=1e-5):
        n_objects, n_features = X.shape
        result = np.zeros((n_objects, 3 * n_features), dtype=self.dtype)
        for fold_idx, fold_counter in self.fold_counters:
            for i, column in enumerate(X.columns):
                for j in fold_idx:
                    value = X.iloc[j, i]
                    mean_target, fraction = fold_counter[column][value]
                    result[j, 3 * i] = mean_target
                    result[j, 3 * i + 1] = fraction
                    result[j, 3 * i + 2] = (mean_target + a) / (fraction + b)
        return result

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    unique_values = np.unique(x)
    enc_x = np.eye(unique_values.shape[0])[x]
    weights = np.zeros(enc_x.shape[1])
    lr = 1e-2

    for i in range(1000):
        p = np.dot(enc_x, weights)
        grad = np.dot(enc_x.T, (p - y))
        weights -= grad * lr

    return weights
