import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_contrast(x):
    contrast = x
    for i in range(0, len(x) - 1):
        contrast[i] = x[i] - x[i + 1]
    return contrast


def plot_example_per_class(x, classes, amount):
    """
    Plots an 'amount' of examples per class.
    :param x: input data
    :param classes: al the labels
    :param amount: times to plot a label
    :return: Nothing, but will plot the digits.
    """
    print_tmp = []
    for i in classes:
        print_tmp = print_tmp + x[i][:amount]
    print_tmp = np.array(print_tmp)
    image_helpers.plot_digits(print_tmp, amount)


def get_digits_per_label(x, y, labels):
    """
    Sorts digits per label.
    :param x: input data
    :param y: outpout labels
    :param labels: classes
    :return: digits per label in a dict.
    """
    digits_per_label = dict(zip(labels, [[]] * len(labels)))
    for elem_x, elem_y in zip(x, y):
        digits_per_label[elem_y] = digits_per_label[elem_y] + [elem_x]
    return digits_per_label


def plot_digits(data, num_cols, shape=(28, 28)):
    """
    Plots digits of data in 'num_cols' number of columns. Default shape = (28, 28)
    :param data: Input data.
    :param num_cols: Number of columns.
    :param shape: Shape to plot in (X, Y). default = (28, 28)
    :return: Nothing, plots the input data.
    """
    num_digits = data.shape[0]
    num_rows = int(num_digits / num_cols)
    for i in range(num_digits):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.axis('off')
        plt.imshow(data[i].reshape(shape), interpolation='nearest', cmap='Greys')
    plt.show()


def get_diferences(x0, x1, labels):
    """
    Calculates the diference between to pictures: x1 - x0[element].
    :param x0: input list of pictures
    :param x1:  input picture
    :param labels:
    :return: absolute difference between x0 and x1
    """
    diferences = []
    for label in labels:
        # print np.array(average_digits[label])
        # average_digits[label].shape = (1, 784)
        # plot_digits(np.array([average_digits[label]]), 1)
        diferences.append(sum(abs(x1 - x0[label])))
    return diferences


def fit(average_x, x, labels):
    """
    Fit the x (digit) ont the average_x (average digit). DONOT USE, is enormous time consuming. But increases accuracy.
    :param average_x: average digit
    :param x: digit
    :param labels: labels in average_x and x
    :return:
    """
    lowest_difference = image_helpers.get_diferences(average_x, x, labels)
    for i in range(len(x)/2):
        x[i] = x[i-1]
        differences = image_helpers.get_diferences(average_x, x, labels)
        if np.argmin(differences) < np.argmin(lowest_difference):
            lowest_difference = differences
    return lowest_difference


def get_digit_parts(data, size_x=784, size=16):
    output = []
    z = np.array(data)
    output += z
    return output
