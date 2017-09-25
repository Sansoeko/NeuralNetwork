import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import network_functions as nf


class NeuralNet:
    def __init__(self, output_size, input_size, amount_hidden_layers, hidden_size):
        # define hidden layers
        self.w = [None] * amount_hidden_layers
        for hidden_layer_num in range(amount_hidden_layers):
            self.w[hidden_layer_num] = np.array([[random.uniform(-1, 1) for _ in range(input_size + 1)] for i in range(hidden_size)])

        # define output layer
        self.wL = np.array([[random.uniform(-1, 1) for _ in range(hidden_size + 1)] for i in range(output_size)])

        # bias to zero
        for hidden_layer_num in range(amount_hidden_layers):
            for i in range(hidden_size):
                self.w[hidden_layer_num][i][0] = 0

        for i in range(output_size):
            self.wL[i][0] = 0

        self.hidden_size = hidden_size
        self.amount_hidden_layers = amount_hidden_layers
        self.input_size = input_size
        self.output_size = output_size
        self.act = nf.activation_function_logistic

    def train(self, x, y_true, labels, num_epochs=20, eta=0.5):
        all_errors = []
        output_targets = [[1 if elem_label == elem_y else 0 for elem_label in labels] for elem_y in y_true]

        for _ in tqdm(range(num_epochs)):
            error_sum = 0
            for elem_x, output_target in tqdm(zip(x, output_targets)):
                # predicting y_true
                output_values, hidden_layer = self.input_to_output(elem_x)

                # calculating error
                last_error = (output_values - output_target)*output_values*(([1]*10) - output_values)
                error_sum += np.sum((output_target-output_values)**2)

                # calculating gradient and new weights
                w_output_new = [None] * self.output_size
                for i in range(self.output_size):
                    new_bias = 0  # np.array([self.wL[i][0] - eta * output_error[i]])      # 1
                    new_weights = np.array(self.wL[i][1:] - eta * last_error[i] * hidden_layer[-1])    # 50
                    w_output_new[i] = np.append(new_bias, new_weights)

                deltas_times_weights = []
                for i in range(self.hidden_size):
                    deltas_times_weights.append(
                        sum(last_error * [self.wL[j][i + 1] for j in range(self.output_size)]))

                w_hidden_tmp = []
                # calculating gradient and new weights for the hidden layers
                for num_hidden_layer in range(self.amount_hidden_layers):
                    w_hidden_new = [None] * self.hidden_size
                    for i in range(self.hidden_size):
                        hidden_error = deltas_times_weights[i] * hidden_layer[-1-num_hidden_layer][i] * (1 - hidden_layer[-1-num_hidden_layer][i])
                        new_weights = np.array(self.w[-1-num_hidden_layer][i][1:] - eta * elem_x * hidden_error)
                        w_hidden_new[i] = np.append(new_bias, new_weights)
                    last_error = hidden_error

                    deltas_times_weights = []
                    for i in range(self.hidden_size):
                        for j in range(self.hidden_size):
                            deltas_times_weights.append(
                                np.sum(last_error * self.w[-1-num_hidden_layer][i][j + 1]))

                    # assign calculated weights to list
                    w_hidden_tmp.append(w_hidden_new)

                # updating weights
                self.w = w_hidden_tmp  # np.array([elem.tolist() for elem in w_hidden_tmp])
                self.wL = w_output_new  # np.array([elem.tolist() for elem in w_hidden_new])
            all_errors.append(error_sum)
        plt.plot(range(len(all_errors)), all_errors)
        print
        print "eta:" + str(eta)
        plt.show()

    def input_to_output(self, elem_of_input):
        hidden_layer = [None] * self.amount_hidden_layers
        for num_layer in range(self.amount_hidden_layers):
            hidden_layer[num_layer] = np.array(
                [self.act(np.sum(self.w[num_layer][i][1:] * elem_of_input + self.w[0][i][0])) for i in range(self.hidden_size)])
        output_values = np.array(
            [self.act(np.sum(self.wL[i][1:] * hidden_layer[-1] + self.wL[i][0])) for i in range(self.output_size)])
        return output_values, hidden_layer

    def predict(self, x):
        y_pred = []
        for elem in tqdm(x):
            # predicting y_true
            y_pred.append(np.argmax(self.input_to_output(elem)[0]))
        return y_pred
