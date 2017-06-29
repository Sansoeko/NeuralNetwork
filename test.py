import gzip, cPickle
import matplotlib.pyplot as plt

def load_mnist():
    f = gzip.open('mnist.pkl.gz', 'rb')
    data = cPickle.load(f)
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
    

if __name__=="__main__":
	(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

	#plot_digits(x_train[0:1], 1)

	print t_train.shape