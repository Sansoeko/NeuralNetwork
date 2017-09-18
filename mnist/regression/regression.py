import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import models.network_functions as nf


def train(x, y_true, labels, num_epochs=75, alpha=0.1):
    w = np.array(([random.uniform(-1, 1) for _ in range(20)]) * len(labels))
    w_hidden = np.array([random.uniform(-1, 1) for _ in range(len(x[0])*len(labels)*2)])
    bias = np.array([random.uniform(-0.1, 0.1)] * len(labels))  # to implement correctly

    for elem_x, elem_y in tqdm(zip(x, y_true)):
        for elem_label in labels:
            for _ in range(num_epochs):
                y_true = 1 if elem_y == elem_label else 0
                w_hidden_output = [(w_hidden[i] * elem_x) for i in range(20)]
                y_pred = nf.activation_function_logistic(np.mean(w[elem_label]*w_hidden_output))
                delta = (y_pred - y_true)*elem_x
                w[elem_label] = w[elem_label] + alpha * delta

    # plt.plot(range(len(errors)), errors)
    # plt.show()
    return w, w_hidden, bias


def predict(x, w, w_hidden, bias, labels):
    y_pred = []
    for elem in tqdm(x):
        args = []
        hidden_output = np.array([nf.activation_function_logistic(np.mean(elem * w_hidden[i_hidden * 784:i_hidden * 784 + 784])) for i_hidden in range(20)])
        for i in range(len(labels)):
            args += [nf.activation_function_logistic(np.mean(w[i*20:i*20+20] * hidden_output))]
        y_pred += [np.argmax(args)]
    # y_pred = [np.argmax(np.mean([w[i] * elem for i in range(len(labels))])) for elem in x]
    return y_pred


def train_regression(x, y_true, labels, num_epochs=75, alpha=0.1):
    w = np.array([[random.uniform(-1, 1) for _ in range(len(x[0]))]] * len(labels))
    bias = np.array([[random.uniform(-1, 1)]] * len(labels))

    for elem_label in tqdm(labels):
        y_tmp = [1 if elem == elem_label else 0 for elem in y_true]
        errors = []
        w_tmp = w[elem_label]
        bias_tmp = bias[elem_label]
        for _ in tqdm(range(num_epochs)):
            error = 0
            for elem_x, elem_y_tmp in tqdm(zip(x, y_tmp)):
                y_pred = nf.activation_function_logistic(np.mean(w_tmp * elem_x) + bias_tmp)
                error += (y_pred - elem_y_tmp)**2
                w_tmp = w_tmp - alpha * (y_pred - elem_y_tmp)*elem_x
                bias_tmp = bias_tmp - alpha * (np.mean((y_pred - elem_y_tmp)*elem_x) + bias_tmp)
            errors.append(error)
        w[elem_label] = w_tmp
        bias[elem_label] = bias_tmp
    return w, bias


def predict_regression(x, w, bias, labels):
    y_pred = []
    for elem in x:
        args = []
        for i in range(len(labels)):
            args += [nf.activation_function_logistic(np.mean(w[i] * elem)+bias[i])]
        y_pred += [np.argmax(args)]
    # y_pred = [np.argmax(np.mean([w[i] * elem for i in range(len(labels))])) for elem in x]
    return y_pred
