import numpy as np

import sklearn
from sklearn.cluster import KMeans


class KMeansClassifier(sklearn.base.BaseEstimator):
    def __init__(self, n_clusters):
        """
        :param int n_clusters: Число кластеров которых нужно выделить в обучающей выборке с помощью алгоритма
        кластеризации
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.cluster_to_label_mapping = None

    def fit(self, data, labels):
        """
            Функция обучает кластеризатор KMeans с заданным числом кластеров, а затем с помощью
        self._best_fit_classification восстанавливает разметку объектов

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов обучающей выборки :param
        :np.ndarray labels: Непустой одномерный массив. Разметка обучающей выборки. Неразмеченные объекты имеют метку
        -1.
            Размеченные объекты могут иметь произвольную неотрицательную метку. Существует хотя бы один размеченный
        объект
        :return KMeansClassifier
        """
        self.kmeans.fit(data)
        cluster_labels = self.kmeans.labels_
        self.cluster_to_label_mapping, _ = self._best_fit_classification(cluster_labels, labels)

        return self

    def predict(self, data):
        """
        Функция выполняет предсказание меток класса для объектов, поданных на вход. Предсказание происходит в два этапа
            1. Определение меток кластеров для новых объектов
            2. Преобразование меток кластеров в метки классов с помощью выученного преобразования

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов
        :return np.ndarray: Предсказанные метки класса
        """
        cluster_labels = self.kmeans.predict(data)
        predictions = self.cluster_to_label_mapping[cluster_labels]

        return predictions

    def _best_fit_classification(self, cluster_labels, true_labels):
        """
        :param np.ndarray cluster_labels: Непустой одномерный массив. Предсказанные метки кластеров.
            Содержит элементы в диапазоне [0, ..., n_clusters - 1]
        :param np.ndarray true_labels: Непустой одномерный массив. Частичная разметка выборки.
            Неразмеченные объекты имеют метку -1. Размеченные объекты могут иметь произвольную неотрицательную метку.
            Существует хотя бы один размеченный объект
        :return
            np.ndarray mapping: Соответствие между номерами кластеров и номерами классов в выборке,
                то есть mapping[idx] -- номер класса для кластера idx
            np.ndarray predicted_labels: Предсказанные в соответствии с mapping метки объектов

            Соответствие между номером кластера и меткой класса определяется как номер класса с максимальным числом
            объектов внутри этого кластера. * Если есть несколько классов с числом объектов, равным максимальному,
            то выбирается метка с наименьшим номером. * Если кластер не содержит размеченных объектов, то выбирается
            номер класса с максимальным числом элементов в выборке. * Если же и таких классов несколько,
            то также выбирается класс с наименьшим номером
        """
        mapping = np.zeros(self.n_clusters, dtype=int)
        predicted_labels = np.zeros_like(cluster_labels, dtype=int)

        for cluster in range(self.n_clusters):
            indices = cluster_labels == cluster
            labels_in_cluster = true_labels[indices]
            labeled_indices = labels_in_cluster != -1

            if labeled_indices.any():
                unique_labels, counts = np.unique(labels_in_cluster[labeled_indices], return_counts=True)
                mapping[cluster] = unique_labels[np.argmax(counts)]
            else:
                unique_labels, counts = np.unique(true_labels[true_labels != -1], return_counts=True)
                mapping[cluster] = unique_labels[np.argmax(counts)]

            predicted_labels[indices] = mapping[cluster]

        return mapping, predicted_labels
