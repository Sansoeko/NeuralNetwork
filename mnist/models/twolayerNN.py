import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import network_functions as nf


class TwoLayerNN:
    def __init__(self, output_size, input_size, hidden_size):
        w0 = np.array([[random.uniform(-1, 1) for _ in range(input_size + 1)] for i in range(hidden_size)])
        w1 = np.array([[random.uniform(-0.1, 0.1) for _ in range(hidden_size + 1)] for i in range(output_size)])
        self.w = [w0, w1]
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.act = nf.activation_function_logistic

    def train(self, x, y_true, labels, num_epochs=3, eta=0.001):
        errors = []
        y_true_onehot = [[1 if elem_label == elem_y else 0 for elem_label in labels] for elem_y in y_true]

        for _ in tqdm(range(num_epochs)):
            error = 0
            for elem_x, elem_y_onehot in tqdm(zip(x, y_true_onehot)):
                # predicting y_true
                hidden_layer = np.array(
                    [[self.act(np.sum(self.w[0][i][1:] * elem_x + self.w[0][i][0]))]for i in range(self.hidden_size)])
                y_output = np.array(
                    [[self.act(np.sum(self.w[1][i][1:] * hidden_layer + self.w[1][i][0]))]for i in range(self.output_size)])

                # calculating error
                error1 = (np.concatenate(y_output) - elem_y_onehot)*np.concatenate(y_output)*(([1]*10) - np.concatenate(y_output)) # np.concatenate(y_output) - elem_y_onehot
                # print np.concatenate(y_output)
                error += np.sum(error1**2)

                # calculating gradient and new weights
                w_output_tmp = [None] * self.output_size
                for i in range(self.output_size):
                    bias = 0  # np.array([self.w[1][i][0] - eta * error1[i]])      # 1
                    weights = np.array(self.w[1][i][1:] - eta * error1[i] * np.concatenate(hidden_layer))    # 20
                    w_output_tmp[i] = np.append(bias, weights)

                vector20 = []
                w_hidden_tmp = [None] * self.hidden_size
                for i in range(self.output_size):
                    vector20.append(error1[i] * self.w[1][i][1:])
                vector20 = np.transpose(vector20)

                for i in range(self.hidden_size):
                    b0 = np.array(
                        [(np.sum(self.w[0][j][1:] + self.w[0][j][0])) for j in range(self.hidden_size)])
                    error0 = np.sum(vector20) * self.act(b0) * (1 - self.act(b0))
                    bias = 0  # np.array([self.w[0][i][0] - eta * error0[i]])    #   1
                    weights = np.array(self.w[0][i][1:] - eta * error0[i] * elem_x)  # 784
                    w_hidden_tmp[i] = np.append(bias, weights)

                # updating weights
                self.w[1] = np.array([elem.tolist() for elem in w_output_tmp])
                self.w[0] = np.array([elem.tolist() for elem in w_hidden_tmp])
            errors.append(error)
        plt.plot(range(len(errors)), errors)
        plt.show()

    def predict(self, x):
        y_pred = []
        for elem in tqdm(x):
            # predicting y_true
            hidden_layer = np.array(
                [[self.act(np.sum(self.w[0][i][1:] * elem + self.w[0][i][0]))] for i in range(self.hidden_size)])
            y_output = np.array(
                [[self.act(np.sum(self.w[1][i][1:] * hidden_layer + self.w[1][i][0]))] for i in
                 range(self.output_size)])
            y_pred.append(np.argmax(y_output))
        return y_pred

