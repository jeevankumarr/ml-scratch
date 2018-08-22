import numpy as np
import random
from tabulate import tabulate
import pandas as pd
from .multivariate_linear_regression import rmse, evaluate_algorithm, get_range, standardize

def predict(row, coefficients):
    yhat = coefficients[0]
    for x, b in zip(row, coefficients[1:]):

        yhat += b * x

    return 1.0 / (1.0 + np.exp(-yhat))

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
            coef[0] -= learning_rate * err * yhat * (1.0 - yhat)

            for j in range(len(row) - 1):
                coef[j + 1] -= learning_rate * err * row[j] * yhat * (1.0 - yhat)
        # print("iter = {0}, l = {1:.3f}, err = {2:.3f}".format(i, learning_rate, sum_err))
    return coef, sum_err

def logistic_regression(train, test, l_rate, n_epoch):
    """Logistic Regression

    Args:
        train:
        test:
        l_rate:
        n_epoch:

    Returns:

    """
    predictions = list()
    coef, _ = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = np.round(predict(row, coef))
        predictions.append(yhat)
    return(predictions)

# Calculate accuracy percentage
def logistic_accuracy(predicted, actual):
    # print("Actual", actual)
    # print("Pred", predicted)
    correct = 0
    for y, yhat in zip(actual, predicted):
        if y == yhat:
            correct += 1
    return correct / float(len(actual)) * 100.0


if __name__ == "__main__":
    # test predictions
    dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1],
               [7.673756466, 3.508563011, 1]]
    l_rate = 0.01
    n_epoch = 5000
    coef = coefficients_sgd(pd.DataFrame(dataset).values, l_rate, n_epoch)
    print(coef)

    pima = pd.read_csv("../data/pima-diabetes.csv")
    print(tabulate(pima.head(), headers="keys", tablefmt="psql"))
    print(tabulate(pima.describe(), headers="keys", tablefmt="psql"))
    df = standardize(pima)
    print(tabulate(df.head(10), headers="keys", tablefmt="psql"))
    err = evaluate_algorithm(df, logistic_regression, 5, 0.01, 15000, logistic_accuracy)
    print(err, np.mean(err))
