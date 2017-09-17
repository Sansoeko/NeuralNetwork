import gzip
import cPickle
import numpy as np
import random

import models.image_helpers as image_helpers
from models.model_average import ModelAverage
from models.model_contrast import ModelContrast
import regression.regression as regression
import evaluate


def load_mnist():
    """
    Loads data. (x_train, y_train), (x_valid, y_valid), (x_test, y_test) from mnist.pkl.gz
    :return: data
    """
    data_file = gzip.open('mnist.pkl.gz', 'rb')
    data = cPickle.load(data_file)
    data_file.close()
    return data


if __name__ == "__main__":
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist()
    labels = range(10)
    train_digits_per_label = image_helpers.get_digits_per_label(x_train, y_train, labels)
    # plot_example_per_class(train_digits_per_label, labels, 10)

    """
    plot_tmp = np.array(get_digit_parts(x_valid[0]))
    plot_digits(plot_tmp, 1, (28, 28))
    exit()
    """

    # x_train = [x_train[i] for i, elem in enumerate(y_train) if elem < 2]
    # y_train = [elem for elem in y_train if elem < 2]

    w, w_hidden, bias = regression.train(x_train, y_train, labels)
    # z = np.array([w])
    # print z
    # image_helpers.plot_digits(z, 5)

    y_pred = regression.predict(x_valid, w, w_hidden, bias, labels)
    accuracies, accuracy = evaluate.get_accuracy(y_pred, y_valid, [0, 1])
    print accuracies
    print "Accuracy: {}%!".format(accuracy * 100.0)
    exit()

    model_average = ModelAverage()
    model_average.train(train_digits_per_label, labels)
    model_average.plot_average_digits()
    y_pred = model_average.predict(x_valid, labels)
    accuracies, accuracy = evaluate.get_accuracy(y_pred, y_valid, labels)
    print accuracies
    print "Accuracy: {}%!".format(accuracy * 100.0)
    exit()

    model_contrast = ModelContrast()
    model_contrast.train(train_digits_per_label, labels)
    # model_contrast.plot_contrast_average_digits()
    y_pred = model_contrast.predict(x_valid, labels)
    accuracies, accuracy = evaluate.get_accuracy(y_pred, y_valid, labels)
    print accuracies
    print "Accuracy: {}%!".format(accuracy * 100.0)


