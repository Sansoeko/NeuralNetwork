import cPickle
import gzip
import argparse

import evaluate
import models.image_helpers as image_helpers
from models.regression import regression as Regression
from models.twolayerNN import TwoLayerNN
from models.neural_net import NeuralNet
from models.model_average import ModelAverage
from models.model_contrast import ModelContrast
from models.model_convulution import ModelConvolve


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_hidden_layers", type=int)
    parser.add_argument("--num_hidden_neurons", type=int)
    FLAGS, _ = parser.parse_known_args()

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist()
    labels = range(10)
    train_digits_per_label = image_helpers.get_digits_per_label(x_train, y_train, labels)
    # plot_example_per_class(train_digits_per_label, labels, 10)

    model_nn = NeuralNet(len(labels), len(x_train[0]), FLAGS.num_hidden_layers, FLAGS.num_hidden_neurons)
    model_nn.train(x_train, y_train, x_valid, y_valid, labels)
    y_pred = model_nn.predict(x_valid)
    accuracies, accuracy = evaluate.get_accuracy(y_pred, y_valid, labels)
    print accuracies
    if accuracy > 0.9:
        print "Accuracy: {}%!".format(accuracy * 100.0)
    else:
        print "Accuracy: {}% ..mehhh...".format(accuracy * 100.0)
    exit()

    model_convulution = ModelConvolve()
    x_train_convuluted = model_convulution.train(x_train, y_train)
    image_helpers.plot_digits(x_train[:25], 5)
    image_helpers.plot_digits(x_train_convuluted[:25], 5)
    exit()

    model_twolayerNN = TwoLayerNN(len(labels), len(x_train[0]), 20)
    model_twolayerNN.train(x_train, y_train, labels)
    y_pred = model_twolayerNN.predict(x_valid)
    accuracies, accuracy = evaluate.get_accuracy(y_pred, y_valid, labels)
    print accuracies
    if accuracy > 0.9:
        print "Accuracy: {}%!".format(accuracy * 100.0)
    else:
        print "Accuracy: {}% ..mehhh...".format(accuracy * 100.0)
    exit()

    model_regression = Regression()
    w, bias = model_regression.train(x_train, y_train, labels)
    y_pred = model_regression.predict(x_valid, w, bias, labels)
    accuracies, accuracy = evaluate.get_accuracy(y_pred, y_valid, [0, 1])
    print accuracies
    print "Accuracy: {}%!".format(accuracy * 100.0)

    model_average = ModelAverage()
    model_average.train(train_digits_per_label, labels)
    model_average.plot_average_digits()
    y_pred = model_average.predict(x_valid, labels)
    accuracies, accuracy = evaluate.get_accuracy(y_pred, y_valid, labels)
    print accuracies
    print "Accuracy: {}%!".format(accuracy * 100.0)

    model_contrast = ModelContrast()
    model_contrast.train(train_digits_per_label, labels)
    # model_contrast.plot_contrast_average_digits()
    y_pred = model_contrast.predict(x_valid, labels)
    accuracies, accuracy = evaluate.get_accuracy(y_pred, y_valid, labels)
    print accuracies
    print "Accuracy: {}%!".format(accuracy * 100.0)
