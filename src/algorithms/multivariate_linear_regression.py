import random

import numpy as np
import pandas as pd
from tabulate import tabulate


# from .simple_linear_regression import rmse

def rmse(a, b):
    """Calculate Root Mean Square Error

    Args:
        a (list): vector of float values
        b (list): vector of float values

    Returns:
        float: RMSE of vectors a, b
    """
    if len(a) != len(b):
        print(len(a), len(b))
        raise ValueError("{0} and {1} must have the same length".format(a, b))

    err = 0.0

    for ai, bi in zip(a, b):
        err += np.power(ai - bi, 2)
    err = err / len(a)
    return np.power(err, 0.5)


def mape(a, b):
    if len(a) != len(b):
        print(len(a), len(b))
        raise ValueError("{0} and {1} must have the same length".format(a, b))

    abs_perct_err = []

    for ai, bi in zip(a, b):
        abs_perct_err.append(np.abs((ai - bi) * 1.0 / ai))

    return np.mean(abs_perct_err)


def predict(row, coefficients):
    yhat = coefficients[0]
    # print(row, coefficients)
    for xi, bi in zip(row, coefficients[1:]):
        yhat += xi * bi

    return yhat


def coefficients_sgd(train, learning_rate, iter):
    """Stochastic Gradient Descent

    Args:
        train (numpy.ndarray): input data
        learning_rate (float): rate of learning
        iter (int): No. of iterations

    Returns:
        list, float:

    """
    coef = [0.0] * len(train[0])
    # print("default", coef)
    sum_err = 0.0
    for i in range(iter):
        sum_err = 0.0
        for row in train:
            yhat = predict(row, coef)
            err = yhat - row[-1]
            sum_err += err ** 2
            coef[0] = coef[0] - learning_rate * err

            for j in range(len(row) - 1):
                coef[j + 1] = coef[j + 1] - learning_rate * err * row[j]
        # print("iter = {0}, l = {1:.3f}, err = {2:.3f}".format(i, learning_rate, sum_err))
    return coef, sum_err


def get_range(dataset):
    """Gets the range for each column in the dataset

    Args:
        dataset (pandas.DataFrame): input dataset

    Returns:
        list: list of min and max values for each column
    """
    min_max = []
    for col in dataset.columns:
        min_max.append([min(dataset[col]), max(dataset[col])])
    return min_max


def standardize(dataset):
    """Standardize the data-set based on min max values of the column

    Args:
        data-set(pandas.DataFrame): input data-set

    Returns:
        df (pandas.DataFrame): standardized output data-frame

    """
    rows = []
    min_max = get_range(dataset)
    for row in dataset.values:
        output_row = []

        for val, mm in zip(row, min_max):
            output_row.append((val - mm[0]) * 1.0 / (mm[1] - mm[0]))
        rows.append(output_row)
    df = pd.DataFrame(rows)
    df.columns = dataset.columns
    return df


def linear_regression_sgd(train, test, learn_rate, iter):
    """

    Args:
        train (pandas.DataFrame): training dataset
        test (pandas.DataFrame): test dataset
        learn_rate (float): learning rate for each iteration
        iter (int): no. of iterations

    Returns:
        list: predictions for the test dataset
    """
    preds = []
    coef, _ = coefficients_sgd(train, learn_rate, iter)
    for row in test:
        yhat = predict(row, coef)
        preds.append(yhat)
    return preds


def cross_validation_split(dataset, n_folds):
    """

    Args:
        dataset (pandas.DataFrame): input dataframe
        folds (int): no. of folds

    Returns:

    """
    dataset_split = []
    dataset_copy = list(dataset.values)
    fold_size = int(len(dataset)) / n_folds

    for _ in range(n_folds):
        fold = []
        while len(fold) < fold_size and len(dataset_copy) > 0:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)

    return dataset_split


def evaluate_algorithm(dataset, algorithm, n_folds, learn_rate, iter, accuracy_metric):
    folds = cross_validation_split(dataset, n_folds)
    scores = []

    for i, fold in enumerate(folds):
        train = [f for j, f in enumerate(folds) if j != i]
        train = sum(train, [])
        test = []
        for row in fold:
            row_copy = list(row)
            test.append(row_copy)
            row_copy[-1] = None
        pred = algorithm(train, test, learn_rate, iter)
        actual = [row[-1] for row in fold]

        err = accuracy_metric(pred, actual)

        scores.append(err)

    return scores


if __name__ == "__main__":
    dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
    # coef = [0.4, 0.8]
    coef, err = coefficients_sgd(dataset, 0.001, 50)
    print("Coefficients: {0}, err {1}".format(coef, err))

    for row in dataset:
        yhat = predict(row, coef)
        print("Expected {0:.4f}, Predicted = {1:.4f}".format(row[-1], yhat))

    wine_data = pd.read_csv("../data/winequality-white.csv", sep=";")
    print(tabulate(wine_data.head(), headers="keys", tablefmt="psql"))
    print(tabulate(wine_data.describe(), headers="keys", tablefmt="psql"))
    # Calcualtion co-efficients on raw data will lead to high value of errors
    min_max = get_range(wine_data)
    print("min max", min_max)
    df = standardize(wine_data)
    print(tabulate(df.head(), headers="keys", tablefmt="psql"))

    # coef, err = coefficients_sgd(df.values, 0.01, 1000)
    # print("Coefficients: {0}, err {1}".format(coef, err))

    errs = evaluate_algorithm(df, linear_regression_sgd, 5, 0.01, 1000, rmse)
    print(errs)
    print("Mean Err: {0:.3f}".format(np.mean(errs)))
