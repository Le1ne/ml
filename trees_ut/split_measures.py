import numpy as np


def evaluate_measures(sample):
    _, counts = np.unique(sample, return_counts=True)
    probs = counts / sum(counts)
    measures = {'gini': float(1 - sum(probs ** 2)), 'entropy': float(-sum(probs * np.log(probs))), 'error': float(1 - max(probs))}
    return measures
