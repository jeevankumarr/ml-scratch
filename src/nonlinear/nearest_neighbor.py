import numpy as np

class NearestNeighborClassifier(object):
    def __init__(self, distance_metric='manhattan'):
        self.distance_metric = distance_metric

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _get_distances(self, X_train, row):
        return np.sum(np.abs(X_train - row), axis=1)

    def predict(self, X_test):

        preds = np.zeros(len(X_test))

        for i, row in enumerate(X_test):
            dists = self._get_distances(self.X_train, row)
            min_dist_idx = np.argmin(dists)
            preds[i] = self.y_train[min_dist_idx]

        return preds

class KNearestNeighbors(NearestNeighborClassifier):
    def __init__(self, k, *args, **kwargs):
        super(KNearestNeighbors, self).__init__(*args, **kwargs)
        self.k = k

    def predict(self, X_test):
        preds = np.zeros(len(X_test))

        for i, row in enumerate(X_test):
            dists = self._get_distances(self.X_train, row)

            min_dist_idxs = np.argsort(dists)[:self.k]
            preds[i] = np.mean(np.take(self.y_train, min_dist_idxs))

        return preds