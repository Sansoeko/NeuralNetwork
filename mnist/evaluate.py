import matplotlib.pyplot as plt
from sklearn import metrics


def get_accuracy(y_pred, y_true, labels):
    """
    Calculates the accuracy of a model.
    :param predicted_y: The models predicted answers
    :param true_y: The true answers of the data set.
    :param labels: Possible values for true_y
    :return: (answers the module gets right) / (lenght of predicted_y) per label.
    """

    stats = metrics.classification_report(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    """
    TP = dict(zip(labels, [0] * len(labels)))
    TN = dict(zip(labels, [0] * len(labels)))
    FP = dict(zip(labels, [0] * len(labels)))
    FN = dict(zip(labels, [0] * len(labels)))
    for elem_pred, elem_true in zip(y_pred, y_true):
        if elem_pred == elem_true:
            TP[elem_pred] = TP[elem_pred] + 1
            for label in labels:
                if label != elem_pred:
                    TN[label] = TN[label] + 1
        else:
            FP[elem_pred] = FP[elem_pred] + 1
            FN[elem_true] = FN[elem_true] + 1
            for label in labels:
                if label != elem_pred and label != elem_true:
                    TN[label] = TN[label] + 1

    accuracies = dict(zip(labels, [0] * len(labels)))
    for label in labels:
        print (TP[label] + TN[label])
        print float(TP[label] + TN[label] + FP[label] + FN[label])
        accuracies[label] = (TP[label] + TN[label]) / float(TP[label] + sum(TN.values()) + FP[label] + FN[label])
    accuracy = (sum(TP.values()) + sum(TN.values())) / float(sum(TP.values()) + sum(TN.values()) + sum(FP.values()) + sum(FN.values()))
    return accuracies, accuracy
    """
    return stats, acc


def plot_regression(y1, y2):
    n = range(10)
    plt.subplot(222)
    plt.bar(n, y1, 0.12)
    plt.subplot(223)
    plt.bar(n, y2, 0.12)
    plt.show()