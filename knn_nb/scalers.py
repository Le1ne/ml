import numpy as np
import typing


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data: np.ndarray) -> None:
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min) / (self.max - self.min)


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray) -> None:
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std
