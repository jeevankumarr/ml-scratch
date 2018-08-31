import numpy as np


def predict(x, w, b):
    output = b
    for xi, wi in zip(x, w):
        output += xi * wi
    return output


def cost(y, y_hat):
    """Cost function

    Args:
        y (list): list of actual values
        y_hat (list): list of predicted values

    Returns:
        float: total cost
    """
    loss = 0.0
    for a, p in zip(y, y_hat):
        loss += (a * np.log(p) + (1-a) * np.log(1-p))

    return (loss*-1.0)/len(y)


if __name__ == "__main__":
    y = [1, 0, 1, 0]
    y_hat = [0.99, 0.01, 0.99, 0.01]
    print(cost(y, y_hat))

    y_hat = [0.01, 0.99, 0.01, 0.99]
    print(cost(y, y_hat))