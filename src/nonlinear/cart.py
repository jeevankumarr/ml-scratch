import numpy as np
import pandas as pd
from tabulate import tabulate
#from src.linear.multivariate_linear_regression import evaluate_algorithm
import random

def gini_index(groups, classes):
    """Calculate the gini index for the given dataset and the classes

    Args:
        groups (iterable): list of groups of data
        classes (list): unique outcome lables

    Returns:
        int: gini index
    """
    n_instances = 1.0 * np.sum([len(group) for group in groups])

    gini = 0.0

    for i, group in enumerate(groups):
        # print("Group ", i, group)
        size = float(len(group))

        if size == 0:
            continue

        score = 0.0

        for val in classes:
            p = [row[-1] for row in group].count(val)*1.0 / size
            score += p**2

        gini += (1.0 - score) * (size / n_instances)
    return gini


def test_split(index, value, dataset):
    l, r = [], []
    for row in dataset:
        if row[index] < value:
            l.append(row)
        else:
            r.append(row)
    return l, r


def get_split(dataset):
    """Get the best split for the current dataset based on the gini index

    Args:
        dataset (list): input dataset

    Returns:
        dict: reference to the node
    """
    vals = list(set(row[-1] for row in dataset))
    b_idx, b_val, b_score, b_groups = 999, 999, 999, None
    gini = 0
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, vals)
            # print("X {0} < {1:.3f} Gini = {2:.3f}, {3:.3f}".format(index + 1, row[index], gini, b_score))
            if gini < b_score:
                b_idx, b_val, b_score, b_groups = index, row[index], gini, groups
    # print("\n", np.round(dataset, 3))
    # print("Got split X_{0} < {1}, {2}\n\n".format(b_idx, b_val, gini))

    return {"index": b_idx, "value": b_val, "groups": b_groups}


# create a terminal node value
## Two conditions to decide when to stop growing a tree
## 1. Max tree depth
## 2. Min node records
## Below function returns the most common class value in a given group
def to_terminal(group):
    """Creates Terminal node

    Args:
        group (list): list of rows

    Returns:
        int: output label for the given group
    """
    outcomes = [row[-1] for row in group]
    m = max(set(outcomes), key=outcomes.count)

    return m


# Recursive Splitting
## The basic idea is to create tree levels, and at each level, the best split
## is chosen. The data in each branch of the children groups is split again.
## Any splitting is predicated on the chosen max tree depth or min node records
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    """Recursively splits the dataset based on gini index at each level.
    And, terminates the branch if max depth is reach or row count is lower
    than min size.

    Args:
        node (dict): node references
        max_depth (int): max depth of the tree
        min_size (int): min size of the terminal node
        depth (int): current depth

    Returns:
        None
    """
    left, right = node["groups"]

    del(node["groups"])
    if not left or not right:
        node["left"] = node["right"] = to_terminal(left + right)
        return

    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(left), to_terminal(right)
        return

    for branch_name, branch in [("left", left), ("right", right)]:

        if len(branch) <= min_size:
            node[branch_name] = to_terminal(branch)
        else:
            node[branch_name] = get_split(branch)
            split(node[branch_name], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    """Build a decision tree

    Args:
        train (list): list of input rows
        max_depth (int): max depth of the tree
        min_size (int): min size of the terminal node

    Returns:
        dict: reference to the root of the tree
    """
    root = get_split(train.values)
    split(root, max_depth, min_size, 1)

    return root


def print_tree(node, depth=0):
    """Prints a tree

    Args:
        node (dict): the node of a given tree
        depth (int): current depth of a tree

    Returns:
        None
    """
    if isinstance(node, dict):
        print("{0} [X_{1} < {2:0.3f}]".format("\t"*depth, node["index"] + 1,
                                              node["value"]))
        print_tree(node["left"], depth+1)
        print_tree(node["right"], depth + 1)
    else:
        print("{0} [{1}]".format("\t"*depth, node))


def predict(node, row):
    """Function to predict the outcome from a decision tree for a row of data

    Args:
        node (dict): tree node with references to its branches and conditionals
        row (list): list of values of the given row

    Returns:
        int: classification label for the given tree and row
    """
    if row[node["index"]] < node["value"]:
        if isinstance(node["left"], dict):
            return predict(node["left"], row)
        else:
            return node["left"]
    else:
        if isinstance(node["right"], dict):
            return predict(node["right"], row)
        else:
            return node["right"]


def decision_tree(train, test, max_depth, min_size):
    """Algorithm that builds the tree using the training and returns the
    predictions for the test set.

    Args:
        train (list): list of rows for training
        test (list):  list of rows for prediction
        max_depth (int): maximum depth of the tree
        min_size (int): minimum no. of rows in the terminal node

    Returns:
        list: predictions for the test dataset
    """
    tree = build_tree(train, max_depth, min_size)
    predictions = []

    for row in test.values:
        prediction = predict(tree, row)
        predictions.append(prediction)

    return predictions

def cross_validation_split(dataset, n_folds):
    """Split data into n_folds for cross validation

    Args:
        dataset (pandas.DataFrame): input dataset
        n_folds (int): no. of folds the data needs to be split

    Returns:
        list: list of dataset
    """

    dataset_split = []
    dataset_copy = list(dataset.values)
    fold_size = int(len(dataset)) / n_folds

    for _ in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            if len(dataset_copy) <= 0:
                break
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        t = pd.DataFrame(fold)
        t.columns = dataset.columns
        dataset_split.append(t)
    return dataset_split

def accuracy_metric(actual, predicted):
    """Accuracy metric is the % of correct answers between actual and
    predicted vectors

    Args:
        actual (iterable): vector of actual results
        predicted (iterable): vector for predict results

    Returns:
        float: % of values correct between actual and predicted
    """
    if len(actual) != len(predicted):
        print(len(actual), len(predicted))
        raise ValueError("Actual and predicted must have same length of values")

    correct = [a==p for a, p in zip(actual, predicted)]
    return sum(correct)*1.0 / len(actual)

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    """Evaluates the algorithm via n_fold cross validation.

    Args:
        dataset (numpy.ndarray): input dataset
        algorithm (callable): algorithm
        n_folds (int): no. of folds for cross validation
        *args: other arguments for the algorithm

    Returns:
        list: list of accuracy scores
    """
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    train, test = None, None
    for i, fold in enumerate(folds):
        train = pd.DataFrame()
        test = pd.DataFrame()
        actual = None
        for j, f in enumerate(folds):
            if j != i:
                train = train.append(fold, ignore_index=True)

            else:
                test = test.append(fold, ignore_index=True)
                actual = list(test.iloc[:, -1])
                test.iloc[:, -1] = None

        preds = algorithm(train, test, *args)
        # actual = [row[-1] for row in fold]

        err = accuracy_metric(preds, actual)

        scores.append(err)

    return scores


if __name__ == "__main__":
    a = [
       [[1, 1], [1, 0]],
       [[1, 1], [1, 0]]
    ]
    print(gini_index(a, [0, 1]))
    b = [
       [[1, 0], [1, 0]],
       [[1, 1], [1, 1]]
    ]
    print(gini_index(b, [0, 1]))

    c = [
       [["A", "B", 0], ["C","D", 0]],
       [["I", "J", 1], ["X", "Y", 1]]
    ]
    print(gini_index(c, [0, 1]))

    # Test getting the best split
    dataset = [[2.771244718, 1.784783929, 0],
               [1.728571309, 1.169761413, 0],
               [3.678319846, 2.81281357, 0],
               [3.961043357, 2.61995032, 0],
               [2.999208922, 2.209014212, 0],
               [7.497545867, 3.162953546, 1],
               [9.00220326, 3.339047188, 1],
               [7.444542326, 0.476683375, 1],
               [10.12493903, 3.234550982, 1],
               [6.642287351, 3.319983761, 1]]
    splitx = get_split(dataset)
    print('Split: [X%d < %.3f]' % ((splitx['index'] + 1), splitx['value']))

    tree = build_tree(pd.DataFrame(dataset), 4, 1)
    print_tree(tree)
    banknote = pd.read_csv("../data/banknote.txt", header=None)
    print(tabulate(banknote.head(), headers="keys", tablefmt="psql"))
    print("Dataset shape:", banknote.shape)

    folds = 5
    max_depth = 5
    min_size = 10
    scores = evaluate_algorithm(banknote, decision_tree, folds, max_depth, min_size)
    print("Scores:", np.round(scores, 3))
