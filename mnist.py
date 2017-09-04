import gzip, cPickle
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
        digits_per_label[elem_y].append(elem_x)
    return digits_per_label


def get_diferences(average_digits, x, labels):
    diferences = []
    for label in labels:
        print np.array(average_digits[label])
        average_digits[label].shape = (1, 784)
        plot_digits(np.array(average_digits[label]), 1)
        diferences.append(sum(abs(x - average_digits[label])))
    print len(diferences)
    return diferences


def get_accuracy(predicted_y,true_y):
    if predicted_y == true_y:
        answers_right[data_pos[pos]] = answers_right[data_pos[pos]] + 1.0
    else:
        answers_wrong[data_pos[pos]] = answers_wrong[data_pos[pos]] + 1.0
    return sum(answers_right) / sum(answers_wrong)


class ModelAverage:
    def __init__(self):
        self.average_digits = {}

    def train(self, digits_per_label, labels):
        self.average_digits = dict(zip(labels, [[]] * len(labels)))
        for elem in labels:
            self.average_digits[elem] = sum(sum(digits_per_label[elem])) / float(len(digits_per_label[elem]))
        exit()

    def predict(self, x):
        differences = get_diferences(self.average_digits, x[0], labels)
        print differences
        # return lowest_difference
        exit()
        return predicted_y



if __name__ == "__main__":
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist()
    labels = range(10)
    train_digits_per_label = get_digits_per_label(x_train, y_train, labels)
    for i in labels:
        print_tmp = train_digits_per_label[i][:10]
        print_tmp = np.array(print_tmp)
        print print_tmp.shape
        plot_digits(print_tmp, 2)
    exit()
    model_average = ModelAverage()
    model_average.train(train_digits_per_label, labels)
    predicted_y = model_average.predict(x_valid)
    print get_accuracy(predicted_y, y_valid)

    """
    plot_digits(x_train[:10], 5) #laat de eerset 10 plaatjes zien

    print x_train.shape #waardes van de pixels
    print t_train.shape #de goede antwoorden

    inputnodes: 28**2 = 784
    hiddenodes: ???
    outputnodes: [0,0,0,0,0,0,0,0,0,0] = 10


    DONE 1 create dict with labels as keys and corresponding digits as values (dict1)
    DONE 3 create dict with average digits (dict2)
    DONE 4 print the 10 average digits

    TODO 2 print the first 10 members of every digit
    """
