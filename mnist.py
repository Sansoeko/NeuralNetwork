import gzip
import cPickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_mnist():
    data_file = gzip.open('mnist.pkl.gz', 'rb')
    data = cPickle.load(data_file)
    data_file.close()
    return data


def plot_digits(data, num_cols, shape=(28, 28)):
    num_digits = data.shape[0]
    num_rows = int(num_digits / num_cols)
    for i in range(num_digits):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.axis('off')
        plt.imshow(data[i].reshape(shape), interpolation='nearest', cmap='Greys')
    plt.show()


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


def get_diferences(average_digits, x, labels):
    diferences = []
    for label in labels:
        # print np.array(average_digits[label])
        # average_digits[label].shape = (1, 784)
        # plot_digits(np.array([average_digits[label]]), 1)
        diferences.append(sum(abs(x - average_digits[label])))
    return diferences


def get_accuracy(predicted_y,true_y,labels):
    answers_right = dict(zip(labels, [0] * len(labels)))
    answers_wrong = dict(zip(labels, [0] * len(labels)))
    for i in range(len(predicted_y)): # TODO use zip instead
        if predicted_y[i] == true_y[i]:
            answers_right[true_y[i]] = answers_right[true_y[i]] + 1.0
        else:
            answers_wrong[true_y[i]] = answers_wrong[true_y[i]] + 1.0
    return sum(answers_right.values()) / float(len(predicted_y))


class ModelAverage:
    def __init__(self):
        self.average_digits = {}

    def train(self, digits_per_label, labels):
        self.average_digits = defaultdict()
        for elem in labels:
            self.average_digits[elem] = sum(digits_per_label[elem]) / float(len(digits_per_label[elem]))

    def predict(self, x):
        predicted_y = []
        for elem in x:
            differences = get_diferences(self.average_digits, elem, labels)
            # plot_digits(np.array([elem]), 1)
            predicted_y += [np.argmin(differences)]
        return predicted_y



if __name__ == "__main__":
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist()
    labels = range(10)
    train_digits_per_label = get_digits_per_label(x_train, y_train, labels)
    # TODO print first ten of every class
    # for i in labels:
    #     print_tmp = train_digits_per_label[i][:10]
    #     print_tmp = np.array(print_tmp)
    #     print print_tmp.shape
    #     plot_digits(print_tmp, 2)
    # exit()
    model_average = ModelAverage()
    model_average.train(train_digits_per_label, labels)
    predicted_y = model_average.predict(x_valid)
    # TODO print all average digits
    print 'accuracy: {}%!'.format(get_accuracy(predicted_y, y_valid,labels) * 100)
    # TODO  add accuracy per label
    # TODO doc strings (see get_digits_per_label) for all functions and all classes.
    """
    inputnodes: 28**2 = 784
    hiddenodes: ???
    outputnodes: [0,0,0,0,0,0,0,0,0,0] = 10
    """
