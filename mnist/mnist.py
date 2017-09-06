#3th party
import gzip
import cPickle
from collections import defaultdict
import numpy as np

#1st and 2nd party
import models.image_helpers as image_helpers
import models.stand_funcs as stand_funcs
from models.model_average import ModelAverage
from models.model_contrast import ModelContrast


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

    image_helpers.get_digit_parts(x_train[777])
    exit()

    model_average = ModelAverage()
    model_average.train(train_digits_per_label, labels)
    model_average.plot_average_digits()
    predicted_y = model_average.predict(x_valid, labels)
    accuracy = stand_funcs.get_accuracy(predicted_y, y_valid, labels).values()
    print accuracy
    print format(sum(accuracy) * 100.0) + "%!"

    model_contrast = ModelContrast()
    model_contrast.train(train_digits_per_label, labels)
    model_contrast.plot_contrast_average_digits()
    predicted_y = model_contrast.predict(x_valid, labels)
    accuracy = stand_funcs.get_accuracy(predicted_y, y_valid, labels).values()
    print accuracy
    print format(sum(accuracy) * 100.0) + "%!"

    # TODO doc strings (see get_digits_per_label) for all functions and all classes.
    """
    inputnodes: 28**2 = 784
    hiddenodes: ???
    outputnodes: [0,0,0,0,0,0,0,0,0,0] = 10
    """
