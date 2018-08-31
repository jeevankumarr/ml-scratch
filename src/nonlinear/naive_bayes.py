"""Naive Bayes

Author: jeevan.kr@gmail.com
References: Machine Learning Algorithms from scratch - Jason Brownlee

P (class | data ) = P(data | class) * P (class) / P(data)
"""

import pandas as pd
import numpy as np
from tabulate import tabulate
from collections import defaultdict
from .cart import accuracy_metric
import random


def split_by_class(dataset, label_col):
    labels = dataset[label_col].unique()
    dataset_split = {}

    for label in labels:
        temp = dataset[dataset[label_col] == label].copy().values
        dataset_split[label] = temp

    return dataset_split

def cross_validation_split(dataset, n_folds):
    """Split data into n_folds for cross validation

    Args:
        dataset (numpy.ndarray): input dataset
        n_folds (int): no. of folds the data needs to be split

    Returns:
        list (pandas.DataFrame): list of pandas dataframes
    """

    dataset_split = []
    dataset_copy = list(dataset.values)
    fold_size = int(len(dataset)) / n_folds
    print("CROSS VALIDATION SPLIT")
    for _ in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            if len(dataset_copy) <= 0:
                break
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(pd.DataFrame(fold, columns=dataset.columns))
    return dataset_split

def summarize(train):
    """

    Args:
        train (pandas.DataFrame): input dataset
        label_col (str): label column name

    Returns:

    """

    labels = (train.iloc[:, -1]).unique()
    summaries = defaultdict(list)

    for label in labels:
        temp = train[train.iloc[:, -1] == label].copy()
        for col in temp.columns:
            summaries[label].append((temp[col].mean(),
                                     np.std(temp[col], ddof=1),
                                     len(temp[col])))

    return summaries


def calculate_probability(x, mean, stdev):
    """Calculate the gaussian probability of x given the mean and stdev

    Args:
        x (float): input values
        mean (float): mean of the distribution
        stdev (float): standard deviation of the distribution

    Returns:
        float: probability of the given input value
    """
    exponent = np.exp(-1.0 * ((x-mean)**2 / (2*(stdev**2))))
    return (1/(np.sqrt(2.0 * np.pi) * stdev)) * exponent


def calculate_class_probability(summaries, row):
    """

    Args:
        summaries(dict):
        row (list):

    Returns:

    """
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = {}

    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]*1.0 / float(total_rows)
        for i in range(len(class_summaries)-1):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i],
                                                                mean,
                                                                stdev)
    return probabilities


def encode(df, label_col, label_index_col):
    labels = list(df[label_col].unique())
    labels.sort()
    df[label_index_col] = df.apply(lambda row: labels.index(row[label_col]), axis=1)

    return df, labels


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

        err = accuracy_metric(preds, actual)

        scores.append(err)

    return scores

def predict(summaries, row):
    """Predicts label using Naive Bayes classifier

    Args:
        summaries (dict): list of summaries for each coloumn. Each summary
            consists of (a) mean, (b) standard deviation and (c) count of values
        row (list): value of each variable in the row

    Returns:

    """
    probs = calculate_class_probability(summaries, row)
    best_label, best_prob = None, -1
    for class_val, prob in probs.items():
        if best_label is None or prob > best_prob:
            best_prob = prob
            best_label = class_val
    return best_label


def naive_bayes(train, test):
    """Naive Bayes classifier derives probabilities from train and returns
    predictions from test.

    Args:
        train (pandas.DataFrame): Training dataset
        test (pandas.DataFrame): Test dataset

    Returns:
        list: predictions for the test dataset
    """
    summaries = summarize(train)
    predictions = []
    for row in test.values:
        output = predict(summaries, row)
        predictions.append(output)
    return predictions


if __name__ == "__main__":
    # Produces probability of the value in a give normal distribution
    # print(calculate_probability(0.0, 1.0, 1.0))
    # print(calculate_probability(1.0, 1.0, 1.0))
    # print(calculate_probability(2.0, 1.0, 1.0))

    # Contrived dataset
    dataset = pd.DataFrame([[3.393533211, 2.331273381, 0],
               [3.110073483, 1.781539638, 0],
               [1.343808831, 3.368360954, 0],
               [3.582294042, 4.67917911, 0],
               [2.280362439, 2.866990263, 0],
               [7.423436942, 4.696522875, 1],
               [5.745051997, 3.533989803, 1],
               [9.172168622, 2.511101045, 1],
               [7.792783481, 3.424088941, 1],
               [7.939820817, 0.791637231, 1]])
    dataset.columns = ["A", "B", "label"]
    summaries = summarize(dataset)
    probabilities = calculate_class_probability(summaries, dataset.values[0])

    # Probabilities for each class
    print(probabilities)

    # Load the Iris dataset
    iris = pd.read_csv("../data/iris.csv")
    # print(tabulate(iris.head(), headers="keys", tablefmt="psql"))
    # print(tabulate(iris.astype(str).groupby(by=["species"]).size()
    #                .reset_index(), headers="keys", tablefmt="psql"))
    iris, iris_label_map = encode(iris, "species", "species_index")
    # print(iris_label_map)
    cols = list(iris.columns[:-2])
    cols.append(iris.columns[-1])
    # Iris dataset
    print(tabulate(iris[cols].head(10), headers="keys", tablefmt="psql"))
    summaries = summarize(iris[cols])

    random.seed(1)
    scores = evaluate_algorithm(iris[cols], naive_bayes, 5)

    # Mean error scores for the 5 fold cross validation
    print(np.round(scores, 3), np.round(np.mean(scores), 3))
