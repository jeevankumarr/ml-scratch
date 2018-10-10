# -*- coding: utf-8 -*-
"""Newton Rhapson method for finding root of a function

"""
from random import random
import math
import pandas as pd


def newton_rhapson_root(f, f_dash, x_init, threshold=0.0001):
    """
    Example:

    Args:
        f (func): function whose root needs to be found
        f_dash (func): function's first order differencial
        x_init (float): initialization value for the root
        threshold (float): minimum threshold for the optimization

    Returns:
        float: root of the given function
    """

    x = x_init
    delta = abs(0.0 - f(x))

    iter = 0
    while delta > threshold:
        print("{0}: x = {1:.4f}, delta = {2:.4f}".format(iter, x, delta))
        x = x - (f(x)*1.0 / f_dash(x))
        delta = abs (0.0 - f(x))
        iter += 1

    # print("iterations {0}".format(i))
    return x


if __name__ == '__main__':
    # Example 1: quadratic
    def f(x): return math.pow(x, 2)*1.0 - 4

    def f_dash(x): return 2.0 * x

    print("x = {0:0.4f}".format(newton_rhapson_root(f, f_dash, random() * 100)))

    # Example 2: Quadratic
    def g(x): return math.pow(x, 2) * 1.0 - 5 * x

    def g_dash(x): return 2.0 * x - 5.0

    print("x = {0:0.4f}".format(newton_rhapson_root(g, g_dash, random() * 100)))

    # Example 3: Fifth power of x
    def f(x):
        return 6 * math.pow(x, 5) - 5 * math.pow(x, 4) - 4 * math.pow(x, 3)  + 3 * math.pow(x, 2)

    def f_dash(x):
        return 30 * math.pow(x, 4)  - 20 * math.pow(x, 3)  - 12 * math.pow(x, 2)  + 6 * x

    print("x = {0:0.4f}".format(newton_rhapson_root(f, f_dash, random() * 100)))
