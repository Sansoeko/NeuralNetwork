# TODO add all functions currently in mnist.py except for load_mnist()
import image_helpers


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
    for part in range(size_x/size):
        z = []
        for y in range(int(size**0.5)):
            for x in range(int(size**0.5)):
                # TODO ???
                z.append(data[pos])
        output.append(z)
    return [output]