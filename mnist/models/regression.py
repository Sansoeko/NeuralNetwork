import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import models.network_functions as nf


class regression:
    def __init__(self):
        self.w = np.array([[random.uniform(-1, 1)]])
        self.bias = np.array([[random.uniform(-1, 1)]])

    def train(self, x, y_true, labels, num_epochs=75, alpha=0.1):
        self.w = np.array([[random.uniform(-1, 1) for _ in range(len(x[0]))]] * len(labels))
        self.bias = np.array([[random.uniform(-1, 1)]] * len(labels))
        for elem_label in tqdm(labels):
            y_tmp = [1 if elem == elem_label else 0 for elem in y_true]
            errors = []
            w_tmp = self.w[elem_label]
            bias_tmp = self.bias[elem_label]
            for _ in tqdm(range(num_epochs)):
                error = 0
                for elem_x, elem_y_tmp in tqdm(zip(x, y_tmp)):
                    y_pred = nf.activation_function_logistic(np.mean(w_tmp * elem_x) + bias_tmp)
                    error += (y_pred - elem_y_tmp)**2
                    w_tmp = w_tmp - alpha * (y_pred - elem_y_tmp)*elem_x
                    bias_tmp = bias_tmp - alpha * (np.mean((y_pred - elem_y_tmp)*elem_x) + bias_tmp)
                errors.append(error)
                self.w[elem_label] = w_tmp
            bias[elem_label] = bias_tmp
        return self.w, bias

    def predict(self, x, w, bias, labels):
        y_pred = []
        for elem in x:
            args = []
            for i in range(len(labels)):
                args += [nf.activation_function_logistic(np.mean(self.w[i] * elem)+self.bias[i])]
            y_pred += [np.argmax(args)]
        # y_pred = [np.argmax(np.mean([w[i] * elem for i in range(len(labels))])) for elem in x]
        return y_pred
