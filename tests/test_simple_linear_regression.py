import random

import math
import numpy as np
import pandas as pd

from src.algorithms import mean, variance, coefficients, covariance


def test_mean():
    assert mean([1, 1]) == 1
    assert mean(range(10)) == 4.5

    assert mean(range(100)) == 49.5

    assert mean([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 5.5


def test_variances():
    assert math.isclose(variance(range(10)), np.var(range(10), ddof=1))

    X = [random.random() for i in range(random.randint(1, 50))]

    assert math.isclose(variance(X), np.var(X, ddof=1))


def test_covariance():
    n = random.randint(1, 50)
    x = [random.random() for i in range(n)]
    y = [random.random() for i in range(n)]

    assert math.isclose(np.cov(x, y)[0, 1], covariance(x, y))

    x = [1, 2, 4, 3, 5]
    y = [1, 3, 3, 2, 5]


def test_coefficients():
    dataset = pd.DataFrame([[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]])

    b0, b1 = coefficients(dataset)
    assert math.isclose(b0, 0.4)
    assert math.isclose(b1, 0.8)


def test_simple_linear_regression():
    pass
