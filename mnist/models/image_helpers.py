import matplotlib.pyplot as plt


def plot_digits(data, num_cols, shape=(28, 28)):
    """
    Plots digits of data in 'num_cols' number of columns. Default shape = (28, 28)
    :param data: Input data.
    :param num_cols: Number of columns.
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
