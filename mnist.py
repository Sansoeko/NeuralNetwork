import gzip
import cPickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_mnist():
    """
    Loads data. (x_train, y_train), (x_valid, y_valid), (x_test, y_test) from mnist.pkl.gz
    :return: data
    """
    data_file = gzip.open('mnist.pkl.gz', 'rb')
    data = cPickle.load(data_file)
    data_file.close()
    return data


def plot_digits(data, num_cols, shape=(28, 28)):
    """
    Plots digits of data in 'num_cols' number of collums. Default shape = (28, 28)
    :param data: Input data.
    :param num_cols: Number of collums.
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
    plot_digits(print_tmp, amount)


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


def get_diferences(x0, x1, labels):
    """
    Calculates the difernce between to pictures: x1 - x0[element].
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


def get_accuracy(predicted_y, true_y, labels):
    """
    Calculates the accuracy of a model.
    :param predicted_y: The models predicted answers
    :param true_y: The true answers of the data set.
    :param labels: Possible values for true_y
    :return: (answers the module gets right) / (lenght of predicted_y) per label.
    """
    answers_right = dict(zip(labels, [0] * len(labels)))
    answers_wrong = dict(zip(labels, [0] * len(labels)))
    answers_accuracy = dict(zip(labels, [0] * len(labels)))
    for elem_predict, elem_true in zip(predicted_y, true_y):
        if elem_predict == elem_true:
            answers_right[elem_true] = answers_right[elem_true] + 1.0
        else:
            answers_wrong[elem_true] = answers_wrong[elem_true] + 1.0
    for elem in answers_accuracy:
        answers_accuracy[elem] = answers_right[elem] / float(len(predicted_y))
    return answers_accuracy


def get_contrast(x):
    contrast = x
    for i in range(0, len(x)-1):
        contrast[i] = x[i] - x[i+1]
    return contrast


def fit(average_x, x, labels):
    """
    Fit the x (digit) ont the average_x (average digit). DONOT USE, is enormous time consuming. But increases accuracy.
    :param average_x: average digit
    :param x: digit
    :param labels: labels in average_x and x
    :return:
    """
    lowest_difference = get_diferences(average_x, x, labels)
    for i in range(len(x)/2):
        x[i] = x[i-1]
        differences = get_diferences(average_x, x, labels)
        if np.argmin(differences) < np.argmin(lowest_difference):
            lowest_difference = differences
    return lowest_difference


def get_digit_parts(data, size_x=784, size=16):
    output = []
    for part in range(size_x/size):
        z = []
        for y in range(int(size**0.5)):
            for x in range(int(size**0.5)):
                # TODO ???
                z.append(data[pos])
        output.append(z)
    return [output]


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

    def predict(self, x):
        """
        Predicts x with self.average_digits
        :param x: Data to predict.
        :return: predicted y values.
        """
        predicted_y = []
        for elem in x:
            differences = get_diferences(self.average_digits, elem, labels)
            # plot_digits(np.array([elem]), 1)
            predicted_y += [np.argmin(differences)]
        return predicted_y

    def plot_average_digits(self):
        """
        Plots the average digits, in collums of lenght 5.
        :return: Nothing
        """
        print_tmp = np.array(self.average_digits.values())
        plot_digits(print_tmp, 5)


class ModelContrast:
    def __init__(self):
        self.average_digits = defaultdict()
        self.average_contrast = defaultdict()

    def train(self, digits_per_label, labels):
        for elem in labels:
            self.average_digits[elem] = sum(digits_per_label[elem]) / float(len(digits_per_label[elem]))
        for elem in self.average_digits:
            self.average_contrast[elem] = get_contrast(self.average_digits[elem])

    def predict(self, x):
        predicted_y = []
        for elem in x:
            digit = get_contrast(elem)
            differences = get_diferences(self.average_contrast, digit, labels)
            # plot_digits(np.array([elem]), 1)
            predicted_y += [np.argmin(differences)]
        return predicted_y

    def plot_contrast_average_digits(self):
        print_tmp = np.array(self.average_contrast.values())
        plot_digits(print_tmp, 5)


if __name__ == "__main__":
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist()
    labels = range(10)
    train_digits_per_label = get_digits_per_label(x_train, y_train, labels)
    # plot_example_per_class(train_digits_per_label, labels, 10)

    """
    plot_tmp = np.array(get_digit_parts(x_valid[0]))
    plot_digits(plot_tmp, 1, (28, 28))
    exit()
    """

    model_contrast = ModelContrast()
    model_contrast.train(train_digits_per_label, labels)
    model_contrast.plot_contrast_average_digits()
    predicted_y = model_contrast.predict(x_valid)
    accuracy = get_accuracy(predicted_y, y_valid, labels).values()
    print accuracy
    print format(sum(accuracy) * 100.0) + "%!"
    exit()

    model_average = ModelAverage()
    model_average.train(train_digits_per_label, labels)
    predicted_y = model_average.predict(x_valid)
    model_average.plot_average_digits()
    accuracy = get_accuracy(predicted_y, y_valid, labels).values()
    print accuracy
    print format(sum(accuracy) * 100.0) + "%!"
    # TODO doc strings (see get_digits_per_label) for all functions and all classes.
    """
    inputnodes: 28**2 = 784
    hiddenodes: ???
    outputnodes: [0,0,0,0,0,0,0,0,0,0] = 10
    """
