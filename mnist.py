import gzip, cPickle
import matplotlib.pyplot as plt

def load_mnist():
    data_file = gzip.open('mnist.pkl.gz', 'rb')
    data = cPickle.load(data_file)
    f.close()
    return data


def plot_digits(data, num_cols, shape=(28,28)):
    num_digits = data.shape[0]
    num_rows = int(num_digits / num_cols)
    for i in range(num_digits):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.axis('off')
        plt.imshow(data[i].reshape(shape), interpolation='nearest', cmap='Greys')
    plt.show()


if __name__ == "__main__":
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

    plot_digits(x_train[:10], 5)

    print x_train.shape #waardes van de pixels
    print t_train.shape #de goede antwoorden

	#inputnodes: 28**2 = 784
	#hiddenodes: ???
	#outputnodes: [0,0,0,0,0,0,0,0,0,0] = 10


    #TODO create dict with labels as keys and corresponding digits as values

    #TODO print the first 10 members of every digit

    #TODO create dict with average digits

    #TODO print the 10 average digits
