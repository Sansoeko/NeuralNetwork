import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import network_functions as nf


class TwoLayerNN:
    def __init__(self, labels, input_size):
        self.w = np.array(([[random.uniform(-1, 1) for _ in range(20)]] * len(labels)))
        self.w_hidden = np.array([[random.uniform(-1, 1) for _ in range(input_size)]]*len(labels)*2)
        self.bias = np.array([random.uniform(-0.1, 0.1)] * len(labels))  # to implement correctly

    def train(self, x, y_true, labels, num_epochs=10, alpha=0.1):
        errors = []

        for _ in tqdm(range(num_epochs)):
            error = 0
            for elem_x, elem_y in tqdm(zip(x, y_true)):
                for elem_label in labels:
                    y_true_ = 1 if elem_y == elem_label else 0
                    w_hidden_output = np.array([nf.activation_function_logistic(
                        np.mean(self.w_hidden[i] * elem_x)) for i in range(len(self.w_hidden))])
                    y_pred = nf.activation_function_logistic(np.mean(self.w[elem_label]*w_hidden_output))
                    delta = (y_true_ - y_pred)
                    error += (y_true_ - y_pred)**2
                    for i in range(len(self.w_hidden)):
                        w_hidden_delta = np.sum(delta * self.w_hidden[i])
                        print w_hidden_delta
                        exit()
                        self.w_hidden[i] = self.w_hidden[i] - alpha * w_hidden_delta*elem_x
                    self.w[elem_label] = self.w[elem_label] - alpha * delta * w_hidden_output
            errors.append(error)

        plt.plot(range(len(errors)), errors)
        plt.show()
        return self.w, self.w_hidden, self.bias

    def predict(self, x, w, w_hidden, bias, labels):
        y_pred = []
        for elem in tqdm(x):
            args = []
            w_hidden_output = np.array([nf.activation_function_logistic(np.mean(w_hidden[i] * elem))
                                        for i in range(len(w_hidden))])
            for i in range(len(labels)):
                args += [nf.activation_function_logistic(np.mean(w[i] * w_hidden_output))]
            y_pred += [np.argmax(args)]
        return y_pred

