import numpy as np
import image_helpers
import stand_funcs


class ModelAverage:
    def __init__(self):
        self.average_digits = {}

    def train(self, digits_per_label, labels):
        """
        Trains model Average
        :param digits_per_label: All digits sorted by label
        :param labels: All labels
        :return: Nothing
        """
        for elem in labels:
            self.average_digits[elem] = sum(digits_per_label[elem]) / float(len(digits_per_label[elem]))

    def predict(self, x, labels):
        """
        Predicts x with self.average_digits
        :param x: Data to predict.
        :param labels: Classes
        :return: A list of all predicted y values.
        """
        predicted_y = []
        for elem in x:
            differences = image_helpers.get_diferences(self.average_digits, elem, labels)
            # plot_digits(np.array([elem]), 1)
            predicted_y += [np.argmin(differences)]
        return predicted_y

    def plot_average_digits(self):
        """
        Plots the average digits, in collums of lenght 5.
        :return: Nothing
        """
        print_tmp = np.array(self.average_digits.values())
        image_helpers.plot_digits(print_tmp, 5)
