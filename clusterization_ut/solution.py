import numpy as np
from sklearn.metrics import pairwise_distances


def silhouette_score(x, labels):
    """
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    """

    dist_matrix = pairwise_distances(x)
    num_clusters, count_clusters = np.unique(labels, return_counts=True)
    len_labels = len(labels)

    if len(num_clusters) <= 1:
        return 0

    masks = np.zeros((len_labels, len(num_clusters))).astype(bool)
    sum_dists = np.zeros((len_labels, len(num_clusters)))
    sizes_clusters = np.zeros(len_labels)

    for i, cluster in enumerate(num_clusters):
        masks[:, i] = labels == cluster
        sum_dists[:, i] = np.sum(dist_matrix[:, labels == cluster], axis=1)
        sizes_clusters[labels == cluster] = np.sum(labels == cluster)

    one_elem_cluster = sizes_clusters == 1
    s = sum_dists[masks]
    s[one_elem_cluster] = 0
    s[~one_elem_cluster] /= (sizes_clusters[~one_elem_cluster] - 1)
    d = np.min(((sum_dists / count_clusters)[~masks]).reshape(len_labels, -1), axis=1)
    d[one_elem_cluster] = 0
    ans = np.zeros(len_labels)
    max_s = np.maximum(s, d)
    np.divide(d - s, max_s, out=ans, where=(max_s != 0))
    sil_score = np.mean(ans)

    return sil_score


def bcubed_score(true_labels, predicted_labels):
    """
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    """

    true_uniq, true_inv, true_count = np.unique(true_labels, return_inverse=True, return_counts=True)
    predicted_uniq, predicted_inv, predicted_count = np.unique(predicted_labels, return_inverse=True,
                                                               return_counts=True)
    true_labels[true_labels == 0] = true_uniq[-1] + 1
    predicted_labels[predicted_labels == 0] = predicted_uniq[-1] + 1
    correctness = np.ones((len(true_labels), len(predicted_labels)))
    correctness[(true_labels / true_labels[:, None]) != 1] = 0
    correctness[(predicted_labels / predicted_labels[:, None]) != 1] = 0
    precision = np.mean(np.sum(correctness, axis=1) / predicted_count[predicted_inv])
    recall = np.mean(np.sum(correctness, axis=1) / true_count[true_inv])
    score = 2 * precision * recall / (precision + recall)

    return score
