import matplotlib.pyplot as plt


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


def plot_regression(y1, y2):
    n = range(10)
    plt.subplot(222)
    plt.bar(n, y1, 0.12)
    plt.subplot(223)
    plt.bar(n, y2, 0.12)
    plt.show()