import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import models.network_functions as nf


def train(x, y_true, labels, num_epochs=750, alpha=0.1):
    w = [random.uniform(-1, 1) for _ in range(len(x[0]))] * len(labels)
    for elem_label in tqdm(labels):
        y_tmp = [1 if elem == elem_label else 0 for elem in y_true]
        errors = []
        w_tmp = w[elem_label]
        for _ in tqdm(range(num_epochs)):
            error = 0
            for elem_x, elem_y_tmp in zip(x, y_tmp):
                y_pred = nf.activation_function_logistic(np.mean(w_tmp * elem_x))
                error += (y_pred - elem_y_tmp)**2
                w_tmp = w_tmp - alpha * (y_pred - elem_y_tmp)*elem_x
            errors.append(error)
        w[elem_label] = w_tmp

    plt.plot(range(len(errors)), errors)
    plt.show()
    return w


def predict(x, w, labels):
    y_pred = []
    for elem in x:
        args = []
        for i in range(len(labels)):
            args += [nf.activation_function_logistic(np.mean(w[i] * elem))]
        y_pred += [np.argmax(args)]
    # y_pred = [np.argmax(np.mean([w[i] * elem for i in range(len(labels))])) for elem in x]
    return y_pred

