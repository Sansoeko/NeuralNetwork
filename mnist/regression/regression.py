import matplotlib.pyplot as plt
import numpy as np
import random


def regression(x, y_true, alpha=0.00001):
    index = []

    for i in range(len(y_true)):
        if y_true[i] > 1:
            index.append(i)

    y_true = np.delete(y_true, index)
    x = np.delete(x, index, 0)

    a = []
    for i in range(784):
        a.append(0.5)
        # a.append(random.uniform(0, 1))

    errors = []
    for i3 in range(len(y_true)):
        y_pred = np.mean(a * x[i3])
        error = (y_pred - y_true[i3])**2
        errors.append(error)
        a = a - alpha * (a*x[i3] - y_true[i3])

    plt.plot(range(len(errors)), errors)
    plt.show()
    return a


def predict(x, a):
    y_pred = []
    for elem in x:
        y_pred.append(int(round(np.mean(a * elem))))
    return y_pred

