import random
from cluster import Cluster
import numpy as np


class KMeans:

    def __init__(self, n_clusters, max_iter):
        self.data = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = []

    def fit(self, data, normalize=True):
        self.data = data
        predict = [0] * len(self.data)

        if normalize:
            self.normalize_data()

        dimensions = len(self.data[0])

        for i in range(self.n_clusters):
            center = [random.random()] * dimensions
            self.clusters.append(Cluster(center))

        iter_no = 0
        not_moves = False
        while iter_no <= self.max_iter and not not_moves:
            for cluster in self.clusters:
                cluster.data = []

            for i in range(len(self.data)):
                cluster_index = self.predict(self.data[i])
                self.clusters[cluster_index].data.append(self.data[i])
                predict[i] = cluster_index

            not_moves = True
            for cluster in self.clusters:
                old_center = cluster.center[:]
                cluster.recalculate_center()

                not_moves = not_moves and cluster.center == old_center

            iter_no += 1
        return predict

    def predict(self, datum):
        min_distance = None
        cluster_index = None
        for index in range(len(self.clusters)):
            distance = euclidean_distance(datum, self.clusters[index].center)
            if min_distance is None or distance < min_distance:
                cluster_index = index
                min_distance = distance

        return cluster_index

    def normalize_data(self):
        cols = len(self.data[0])

        for col in range(cols):
            column_data = []
            for row in self.data:
                column_data.append(row[col])

            mean = np.mean(column_data)
            std = np.std(column_data)

            for row in self.data:
                row[col] = (row[col] - mean) / std

    def sum_squared_error(self):
        sse = 0
        for cluster in self.clusters:
            for datum in cluster.data:
                sse += euclidean_distance(cluster.center, datum)

        return sse**2


def euclidean_distance(x, y):
    sq_sum = 0
    for xi, yi in zip(x, y):
        sq_sum += (yi - xi) ** 2
    return sq_sum ** 0.5
