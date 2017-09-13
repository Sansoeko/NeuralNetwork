import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random


def train(x, y_true, num_epochs=75, alpha=0.00001):
    w = [random.uniform(-1, 1) for _ in range(len(x[0]))]
    y_true = [-1000 if elem == 0 else 1000 for elem in y_true]

    errors = []
    for _ in tqdm(range(num_epochs)):
        error = 0
        for elem_x, elem_y_true in zip(x, y_true):
            y_pred = np.mean(w * elem_x)
            error += (y_pred - elem_y_true)**2
            w = w - alpha * (w*elem_x - elem_y_true)*elem_x
        errors.append(error)

    plt.plot(range(len(errors)), errors)
    plt.show()
    return w


def predict(x, a):
    y_pred = [0 if np.mean(a * elem) < 0 else 1 for elem in x]
    return y_pred

