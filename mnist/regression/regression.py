import matplotlib.pyplot as plt
import numpy as np
import random


def regression(x, y_true, alpha=0.1):
    index = []

    for i in range(len(y_true)):
        if y_true[i] > 1:
            index.append(i)

    y_true = np.delete(y_true, index)
    x = np.delete(x, index, 0)

    a = []
    for i in range(784):
        a.append(random.uniform(-1, 1))

    errors = []
    for i in range(30):
        print str(i + 1) + "/30"
        for i in range(len(y_true)):
            y_pred = np.mean(a * x[i])
            error = (y_pred - y_true[i])**2
            errors.append(error)
            for i2 in range(len(a)):
                a[i2] = a[i2] - alpha * (y_pred - y_true[i])

    plt.plot(range(len(errors)), errors)
    plt.show()
    return a





