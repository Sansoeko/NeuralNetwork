import numpy as np
from collections import defaultdict

import image_helpers


class ModelContrast:
    def __init__(self):
        self.average_digits = defaultdict()
        self.average_contrast = defaultdict()

    def train(self, digits_per_label, labels):
        """
        Predicts x with self.average_digits
        :param x: Data to predict.
        :param labels: Classes
        :return: predicted y values in self.average_contrast.
        """
        for elem in labels:
            self.average_digits[elem] = sum(digits_per_label[elem]) / float(len(digits_per_label[elem]))
        for elem in self.average_digits:
            self.average_contrast[elem] = image_helpers.get_contrast(self.average_digits[elem])

    def predict(self, x, labels):
        """
        Uses self.average_contrast to predict y.
        :param x:
        :param labels: Classes
        :return: A list of all predicted y values.
        """
        predicted_y = []
        for elem in x:
            digit = image_helpers.get_contrast(elem)
            differences = image_helpers.get_differences(self.average_contrast, digit, labels)
            # plot_digits(np.array([elem]), 1)
            predicted_y += [np.argmin(differences)]
        return predicted_y

    def plot_contrast_average_digits(self):
        """
        Plots the contrast of the average digit.
        :return: Nothing
        """
        print_tmp = np.array(self.average_contrast.values())
        image_helpers.plot_digits(print_tmp, 5)