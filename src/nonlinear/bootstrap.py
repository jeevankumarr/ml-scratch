"""

"""
import numpy as np
import random
import pandas as pd
from .cart import cross_validation_split, evaluate_algorithm, accuracy_metric, \
    test_split, gini_index, get_split, to_terminal, split, build_tree, predict
from tabulate import tabulate


def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

def bagging(train, test, max_depth, min_size, sample_size, n_trees):
    trees = []
    for _ in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test.values]

    return predictions

def subsample(dataset, ratio=1.0):
    """Sample from a dataset

    Args:
        dataset (pandas.DataFrame): input dataset
        ratio (float): probability of re-sampling

    Returns:

    """
    sample = []
    sample_ct = np.round(len(dataset) * ratio, 0)
    while len(sample) < sample_ct:
        index = random.randrange(len(dataset))
        sample.append((dataset.values[index]))
    return pd.DataFrame(sample, columns=dataset.columns)

if __name__ == "__main__":
    dataset = pd.DataFrame([[random.randrange(10) for i in range(50)]]).T
    for size in [1, 10, 100]:
        sample_means = []
        ratio = 0.1
        for i in range(size):
            sample = subsample(dataset, ratio)
            sample_mean = np.mean(sample.iloc[:,0])
            sample_means.append(sample_mean)
        print("Samples = {0}, Estimated Mean {1}, Actual Mean {2}"
              .format(size, np.mean(sample_means), np.mean(dataset.iloc[:,0])))

    sonar = pd.read_csv("../data/sonar.csv")
    # print(tabulate(sonar.head(), headers="keys", tablefmt="psql"))
    for n_trees in [1, 5, 10, 50]:
        scores = evaluate_algorithm(sonar, bagging, 5, 6, 2, 0.5, n_trees)
        print("Trees {0}, Mean Acc = {1:.3f}, Scores = {2}"
              .format(n_trees, np.mean(scores), scores))