import pandas as pd
import numpy as np
from tabulate import tabulate
import warnings
from .multivariate_linear_regression import coefficients_sgd as train_weights
from .multivariate_linear_regression import linear_regression_sgd as perceptron
from .multivariate_linear_regression import evaluate_algorithm
from .logistic_regression import logistic_accuracy as accuracy_metric

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def predict(row, weights):
    """Predict using perceptron

    Args:
        row (list): list of values
        weights (list): list of weights

    Returns:
        float: return value
    """
    activation = weights[0]
    # print("r", row)
    for x, w in zip(row, weights[1:]):
        activation += x * w

    return 1.0 if activation >= 0.0 else 0.0


if __name__ == "__main__":
    dataset = [
        [2.7810836,2.550537003,0],
        [1.465489372,2.362125076,0],
        [3.396561688,4.400293529,0],
        [1.38807019,1.850220317,0],
        [3.06407232,3.005305973,0],
        [7.627531214,2.759262235,1],
        [5.332441248,2.088626775,1],
        [6.922596716,1.77106367,1],
        [8.675418651,-0.242068655,1],
        [7.673756466,3.508563011,1]
    ]

    weights = [-0.1, 0.2065364014, -0.2341811771]
    dataset = pd.DataFrame(dataset)
    for row in dataset.values:
        print("Expected {0}, Predicted {1}".format(row[-1], predict(row, weights)))

    l_rate = 0.1
    n_epoch = 5
    weights = train_weights(dataset.values, l_rate, n_epoch, predict)
    print(weights)

    # Sonar dataset
    sonar = pd.read_csv("../data/sonar.csv")
    print(tabulate(sonar.iloc[:,0:16].head(), headers="keys", tablefmt="psql"))

    scores = evaluate_algorithm(sonar, perceptron, 3, 0.01, 50000, accuracy_metric, pred=predict)
    print(scores, np.mean(scores))