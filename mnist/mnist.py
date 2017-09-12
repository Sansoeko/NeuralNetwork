import gzip
import cPickle
import numpy as np

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

    a = regression.regression(x_train, y_train)
    image_helpers.plot_digits(np.array([a]), 1)

    index = []
    for i in range(len(y_valid)):
        if y_valid[i] > 1:
            index.append(i)

    y_valid = np.delete(y_valid, index)
    x_valid = np.delete(x_valid, index, 0)

    index = []
    for i in range(len(y_test)):
        if y_test[i] > 1:
            index.append(i)

    y_test = np.delete(y_test, index)
    x_test = np.delete(x_test, index, 0)

    predicted_y = regression.predict(x_valid, a)
    accuracy = evaluate.get_accuracy(predicted_y, y_valid, [0, 1]).values()
    print accuracy
    print format(sum(accuracy) * 100.0) + "%!"

    predicted_y = regression.predict(x_test, a)
    accuracy = evaluate.get_accuracy(predicted_y, y_test, [0, 1]).values()
    print accuracy
    print format(sum(accuracy) * 100.0) + "%!"
    exit()

    model_average = ModelAverage()
    model_average.train(train_digits_per_label, labels)
    # model_average.plot_average_digits()
    predicted_y_average = model_average.predict(x_valid, labels)
    accuracy_average = evaluate.get_accuracy(predicted_y_average, y_valid, labels).values()
    print accuracy_average
    print format(sum(accuracy_average) * 100.0) + "%!"

    model_contrast = ModelContrast()
    model_contrast.train(train_digits_per_label, labels)
    # model_contrast.plot_contrast_average_digits()
    predicted_y_contrast = model_contrast.predict(x_valid, labels)
    accuracy_contrast = evaluate.get_accuracy(predicted_y_contrast, y_valid, labels).values()
    print accuracy_contrast
    print format(sum(accuracy_contrast) * 100.0) + "%!"
