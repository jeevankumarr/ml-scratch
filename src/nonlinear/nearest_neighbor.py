import numpy as np
import pandas as pd
import sklearn.metrics as sk_met
from tabulate import tabulate

class NearestNeighborClassifier(object):
    def __init__(self, distance_metric='manhattan'):
        self.distance_metric = distance_metric

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _get_distances(self, X_train, row):
        if self.distance_metric == 'manhattan':
            return np.sum(np.abs(X_train - row), axis=1)
        elif self.distance_metric == 'euclidean':
            return np.power(np.sum(np.power(X_train - row, 2), axis=1), 0.5)

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
        # print('KNN Pred')
        preds = np.zeros(len(X_test))

        for i, row in enumerate(X_test):
            dists = self._get_distances(self.X_train, row)
            # select k closest neighbors
            min_dist_idxs = np.argsort(dists)[:self.k]

            # take mean of the k closest neighbors
            preds[i] = np.mean(np.take(self.y_train, min_dist_idxs))

        return preds

if __name__ == '__main__':
    np.random.seed(0)
    x1 = np.random.randint(low=0, high=20, size=100)
    x2 = np.random.normal(0, 1, size=len(x1))
    x3 = np.random.normal(0, 1, size=len(x1))
    y = np.random.randint(0, 2, size=len(x1))

    df = pd.DataFrame({'x1':x1, 'x2':x2, 'x3':x3, 'y':y})
    X_train, y_train = df.iloc[:-20, :-1].values, df.iloc[:-20, -1:].values
    X_test, y_test = df.iloc[-20:, :-1].values, df.iloc[-20:, -1:].values

    nn = NearestNeighborClassifier()
    nn.train(X_train, y_train)
    preds = nn.predict(X_test)
    results = pd.DataFrame(sk_met.classification_report(y_test, preds, output_dict=True))
    print(tabulate(results.T, headers='keys', tablefmt='psql'))

    knn = KNearestNeighbors(6)
    knn.train(X_train, y_train)
    preds = knn.predict(X_test)
    results = pd.DataFrame(sk_met.classification_report(y_test, preds.round(), output_dict=True))
    print(tabulate(results.T, headers='keys', tablefmt='psql'))

    knn = KNearestNeighbors(4, distance_metric='euclidean')
    knn.train(X_train, y_train)
    preds = knn.predict(X_test)
    results = pd.DataFrame(sk_met.classification_report(y_test, preds.round(), output_dict=True))
    print(tabulate(results.T, headers='keys', tablefmt='psql'))