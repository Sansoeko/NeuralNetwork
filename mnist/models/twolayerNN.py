import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import network_functions as nf


class TwoLayerNN:
    def __init__(self, output_size, input_size, hidden_size):
        w0 = np.array([[random.uniform(-0.1, 0.1) for _ in range(input_size + 1)] for _ in range(hidden_size)])
        w1 = np.array([[random.uniform(-0.01, 0.01) for _ in range(hidden_size + 1)] for _ in range(output_size)])
        self.w = [w0, w1]
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.act= nf.activation_function_logistic

    def train(self, x, y_true, labels, num_epochs=12, alpha=0.00001):
        errors = []
        y_true_onehot = [[1 if elem_label == elem_y else 0 for elem_label in labels] for elem_y in y_true]
        data = zip(x, y_true_onehot)

        for _ in tqdm(range(num_epochs)):
            error = 0
            random.shuffle(data)
            for elem_x, elem_y_onehot in tqdm(data):
                # predicting y_true
                hidden_layer = np.array(
                    [[self.act(np.sum(self.w[0][i][1:] * elem_x) + self.w[0][i][0])]for i in range(self.hidden_size)])
                y_pred_onehot = np.array(
                    [[self.act(np.sum(self.w[1][i][1:] * hidden_layer) + self.w[1][i][0])]for i in range(self.output_size)])

                # calculating error
                error1 = y_pred_onehot[0] - elem_y_onehot
                error += np.sum(error1*error1)

                # calculating gradient and new weights
                w1_tmp = [None] * self.output_size
                for i in labels:
                    bias = 0 # np.array([self.w[1][i][0] - alpha * error1[i]])      # 1
                    weights = np.array(self.w[1][i][1:] - alpha * error1[i])    # 20
                    w1_tmp[i] = np.append(bias, weights)

                w0_tmp = [None] * self.hidden_size
                for i in range(self.output_size):
                    vector20 = error1[i] * self.w[1][i][1:]

                for i in range(self.hidden_size):
                    b0 = np.array(
                        [(np.sum(self.w[0][j][1:] * elem_x) + self.w[0][j][0]) for j in range(self.hidden_size)])
                    error0 = vector20 * self.act(b0) * (1 - self.act(b0))
                    bias = 0 # np.array([self.w[0][i][0] - alpha * error0[i]])    #   1
                    weights = np.array(self.w[0][i][1:] - alpha * error0[i])  # 784
                    w0_tmp[i] = np.append(bias, weights)

                # updating weights
                self.w[1] = np.array([elem.tolist() for elem in w1_tmp])
                self.w[0] = np.array([elem.tolist() for elem in w0_tmp])
            errors.append(error)
        plt.plot(range(len(errors)), errors)
        plt.show()

    def predict(self, x):
        y_pred = []
        for elem in tqdm(x):
            # predicting y_true
            hidden_layer = np.array(
                [[self.act(np.sum(self.w[0][i][1:] * elem) + self.w[0][i][0])] for i in range(self.hidden_size)])
            y_pred_onehot = np.array(
                [[self.act(np.sum(self.w[1][i][1:] * hidden_layer) + self.w[1][i][0])] for i in
                 range(self.output_size)])
            y_pred.append(np.argmax(y_pred_onehot))
        return y_pred

