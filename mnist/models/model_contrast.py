import numpy as np
from collections import defaultdict

import stand_funcs
import image_helpers

class ModelContrast:
    def __init__(self):
        self.average_digits = defaultdict()
        self.average_contrast = defaultdict()

    def train(self, digits_per_label, labels):
        for elem in labels:
            self.average_digits[elem] = sum(digits_per_label[elem]) / float(len(digits_per_label[elem]))
        for elem in self.average_digits:
            self.average_contrast[elem] = stand_funcs.get_contrast(self.average_digits[elem])

    def predict(self, x, labels):
        predicted_y = []
        for elem in x:
            digit = stand_funcs.get_contrast(elem)
            differences = image_helpers.get_diferences(self.average_contrast, digit, labels)
            # plot_digits(np.array([elem]), 1)
            predicted_y += [np.argmin(differences)]
        return predicted_y

    def plot_contrast_average_digits(self):
        print_tmp = np.array(self.average_contrast.values())
        image_helpers.plot_digits(print_tmp, 5)