from .bootstrap import subsample, bagging_predict
from .cart import gini_index, print_tree
import random
import pandas as pd
import numpy as np
from tabulate import tabulate
from .cart import evaluate_algorithm


def get_split(dataset, n_features):
    """This method randomly picks n_features and returns the best split across
    all the features

    Args:
        dataset(numpy.ndarray): input dataset
        n_features (int): no. of features to be considered for splitting

    Returns:
        dict: of split parameters
    """
    class_values = list(set(row[-1] for row in dataset))
    b_idx, b_val, b_score, b_groups = 999, 999, 999, None

    indicies = random.sample(range(len(dataset[0]) - 1), n_features)
    features = indicies
    # print("features:", features)
    for feature in features:
        for row in dataset:
            groups = split_data(dataset, feature, row[feature])
            gini = gini_index(groups, class_values)

            if gini < b_score:
                b_idx, b_val, b_score, b_groups = feature, row[feature], gini, groups

    return {"index": b_idx, "value": b_val, "groups": b_groups}


def split_data(dataset, feature_index, value):
    """Split the dataset at the given value

    Args:
        dataset (numpy.ndarray): input dataset
        feature_index (int): feature index on which the dataset will be split
        value (float): the value of the threshold

    Returns:
        tuple: groups of dataset
    """

    l, r = [], []
    for row in dataset:
        if row[feature_index] < value:
            l.append(row)
        else:
            r.append(row)
    return l, r


def to_terminal(dataset):
    """Create a terminal node from the group

    Args:
        dataset (numpy.ndarray): input dataset

    Returns:
        int: most popular class label in the dataset
    """
    outcomes = [row[-1] for row in dataset]
    return max(set(outcomes), key=outcomes.count)


def _build_forest(node, max_depth, min_size, n_features, depth):
    """Build the random forest from the current root with the given parameters

    Args:
        node (dict): node of a tree
        max_depth (int): max depth of a tree
        min_size (int): min size of each terminal dataset
        n_features (int): no. of features in each tree
        depth (int): current depth

    Returns:
        None
    """
    # outcomes = [row[-1] for row in group]
    l, r = node["groups"]
    del(node["groups"])

    # check for no split
    if not l or not r:
        node["left"] = node["right"] = to_terminal(l + r)
        return

    # check for max depth
    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(l), to_terminal(r)
        return

    # process left child
    if len(l) <= min_size:
        node["left"] = to_terminal(l)
    else:
        node["left"] = get_split(l, n_features)
        _build_forest(node["left"], max_depth, min_size, n_features, depth + 1)

    # process right child
    if len(r) <= min_size:
        node["right"] = to_terminal(r)
    else:
        node["right"] = get_split(r, n_features)
        _build_forest(node["right"], max_depth, min_size, n_features, depth + 1)


def build_forest(train, max_depth, min_size, n_features):
    """Build a tree

    Args:
        train (pandas.DataFrame): training dataset
        max_depth (int): max depth of the tree
        min_size (int): min size of samples in terminal node
        n_features (int): no. of features for each tree

    Returns:
        dict: tree with left and right sub-trees
    """
    root = get_split(train.values, n_features)
    _build_forest(root, max_depth, min_size, n_features, 1)
    return root


def random_forest(train, test, max_depth, min_size, sample_ratio, tree_ct,
                  feature_ct):
    """Build a random forest from the training dataset and predict outcomes for
    the test dataset

    Args:
        train (pandas.DataFrame): training dataset
        test (pandas.DataFrame): test dataset
        max_depth (int): max depth of any tree
        min_size (int): min size of the dataset at the terminal node
        sample_ratio (float): ratio of re-sampling
        tree_ct (int): no. of trees in the foredst
        feature_ct (int): no. of features for each tree

    Returns:
        (list): list of predictions
    """
    trees = []

    for _ in range(tree_ct):
        sample = subsample(train, sample_ratio)
        tree = build_forest(sample, max_depth, min_size, feature_ct)
        # print_tree(tree)
        trees.append(tree)

    predictions = [bagging_predict(trees, row) for row in test.values]
    return predictions


if __name__ == "__main__":
    sonar = pd.read_csv("../data/sonar.csv")
    print(tabulate(sonar.head(), headers="keys", tablefmt="psql"))
    max_depth = 2
    min_size = 10
    sample_size = 1.0

    feature_count = int(np.sqrt(sonar.shape[1]))
    scores = []
    for n_trees in [1, 5, 10]:
        scores = evaluate_algorithm(sonar, random_forest, 5, max_depth,
                                    min_size, sample_size, n_trees,
                                    feature_count)

    print(np.mean(scores), scores)