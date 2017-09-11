import matplotlib.pyplot as plt
import numpy as np
import random


def regression(x, y_true, alpha=0.01):
    x = x[:3000]
    y_true = y_true[:3000]

    for i in range(len(y_true)):
        if not y_true[i] == 0:
            y_true[i] = 1

    a = []
    for i in range(784):
        a.append(random.uniform(-1, 1))

    errors = []
    for i in range(len(y_true)):
        y_pred = np.mean(a * x)
        error = np.mean((y_pred - y_true)**2)
        errors.append(error)
        a = a - alpha * np.mean((y_pred - y_true))

    plt.plot(range(len(errors)), errors)
    plt.show()
    print y_true
    print a
    print y_pred
    return a





