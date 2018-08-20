import pandas as pd
from tabulate import tabulate


def predict(row, coefficients):
    yhat = coefficients[0]

    for xi, bi in zip(row, coefficients[1:]):
        yhat += xi * bi

    return yhat


def coefficients_sgd(train, learning_rate, iter):
    """Stochastic Gradient Descent

    Args:
        train (list): input data
        learning_rate (float): rate of learning
        iter (int): No. of iterations

    Returns:

    """
    coef = [0.0] * len(train[0])

    for i in range(iter):
        sum_err = 0.0
        for row in train:
            yhat = predict(row, coef)
            err = yhat - row[-1]
            sum_err += err ** 2
            coef[0] = coef[0] - learning_rate * err

            for j in range(len(row) - 1):
                coef[j + 1] = coef[j + 1] - learning_rate * err * row[j]
        print("iter = {0}, l = {1:.3f}, err = {2:.3f}".format(i, learning_rate, sum_err))
    return coef


if __name__ == "__main__":
    dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
    # coef = [0.4, 0.8]
    coef = coefficients_sgd(dataset, 0.001, 50)
    print(coef)
    for row in dataset:
        yhat = predict(row, coef)
        print("Expected {0:.4f}, Predicted = {1:.4f}".format(row[-1], yhat))

    wine_data = pd.read_csv("../data/winequality-white.csv", sep=";")
    print(tabulate(wine_data.head(), headers="keys", tablefmt="psql"))
