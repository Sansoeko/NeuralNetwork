import matplotlib.pyplot as plt
import numpy as np
import random

if __name__ == '__main__':
    x = np.linspace(0, 9, 10)
    # y_true = 3 * x
    y_true = np.random.normal(3 * x, 5.0)

    alpha = 0.01
    a = random.uniform(-10, 10)
    errors = []
    for i in range(30):
        y_pred = a * x
        error = np.mean((y_pred - y_true)**2)
        errors.append(error)
        a = a - alpha * np.mean((y_pred - y_true) * x)
        plt.scatter(x, y_true, zorder=1)
        plt.plot(x, y_pred, zorder=1)
    plt.show()

    plt.plot(range(30), errors)
    plt.show()




