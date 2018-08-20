"""Simple Linear Regression relationship:
        Y = b_0 + b_1 * X

    Author: Jeevan Kumar R (jeevan.kr@gmail.com)
    References: Machine Learning Algorithms From Scratch - Jason Brownlee

    Dataset: https://www.math.muni.cz/~kolacek/docs/frvs/M7222/data/AutoInsurSweden.txt
"""

import random

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as po
from tabulate import tabulate


def mean(values):
    """Return mean of a list"""
    return sum(values) * 1.0 / len(values)


def variance(values):
    """Return the variance of a list of values

    Args:
        values (list): list of input numbers

    Returns:
        float: variance of values
    """

    m = mean(values)

    return sum([np.power(v - m, 2) for v in values]) / (len(values) - 1)


def covariance(x, y):
    """Calculate co-variance of two vectors

    Args:
        x (list): list of numbers
        y (list): list of number

    Returns:
        float: co-variance of x and y

    """

    if len(x) != len(y):
        raise ValueError("x and y must have equal length")

    covar = 0.0

    x_mean = mean(x)
    y_mean = mean(y)

    for xi, yi in zip(x, y):
        covar += (xi - x_mean) * (yi - y_mean)

    return covar * 1.0 / (len(x) - 1)


def coefficients(dataframe):
    """

    Args:
        dataframe (pd.DataFrame): Input dataframe

    Returns:

    """
    x, y = dataframe.T.values
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, y) / variance(x)
    b0 = y_mean - b1 * x_mean

    return b0, b1


def simple_linear_regression(train, test):
    predictions = []
    b0, b1 = coefficients(train)
    print("Coefficients b0={0}, b1={1}".format(b0, b1))

    for x in test:
        yhat = b0 + b1 * x
        predictions.append(yhat)

    return predictions


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


def test_train_split(dataset, split):
    """Split data into test and train

    Args:
        dataset (pandas.DataFrame): DataFrame of the dataset
        split (float): split value of the proportion of size of training set

    Returns:
        (pandas.DataFrame, pandas.DataFrame): training and test datasets
    """
    train, test = [], []
    data = dataset.values
    for row in data:
        if random.random() <= split:

            train.append(list(row))

        else:
            test.append(list(row))
    return pd.DataFrame(train), pd.DataFrame(test)


def evaluate_algorithm(data, algorithm: staticmethod, split=0.8):
    """

    Args:
        data (pd.DataFrame):
        algorithm (callable): Reference to the definition of the algorithm
        split(float): proprtion of len of test to that of the full dataset

    Returns:
        float: Root Mean Square Error
    """
    train, test = test_train_split(data, split)

    test_data = test.T.values[0]
    predicted = algorithm(train, test_data)
    actual = test.T.values[1]
    err = rmse(actual, predicted)

    return err


def plot(X, Y, Y_HAT, title=None):
    traces = []
    traces.append(go.Scatter(x=X, y=Y, mode="markers"))
    traces.append(go.Scatter(x=X, y=Y_HAT, mode="lines"))
    layout = None
    if title is not None:
        layout = go.Layout(title=title)
    fig = go.Figure(data=traces, layout=layout)
    po.plot(fig)


def run():
    random.seed(1)

    # contrived dataset
    print("Contrive a Dataset")
    df = pd.DataFrame([[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]])
    print(tabulate(df, headers='keys', tablefmt='psql'))
    err = evaluate_algorithm(df, simple_linear_regression, split=0.8)
    print(err)
    print("RMSE: {0:.4f}".format(err))

    # swedish insurance dataset
    print("Reading Auto Insurance Dataset")
    auto = pd.read_csv("../data/AutoInsurSweden.txt", sep="\t", skiprows=10)
    auto["X"] = auto["X"] * 1.0
    auto["Y"] = pd.to_numeric(auto.apply(lambda r: r["Y"].replace(",", "."), axis=1))

    print(tabulate(auto.head(), headers="keys", tablefmt="psql"))
    print(auto.dtypes)

    err = evaluate_algorithm(auto, simple_linear_regression, split=0.6)
    print("RMSE: {0:.4f}".format(err))
    pred = simple_linear_regression(auto, auto["X"])
    plot(auto["X"], auto["Y"], pred, title="Swedish Auto Insurance Dataset")


if __name__ == "__main__":
    run()
