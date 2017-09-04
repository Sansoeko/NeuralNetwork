import gzip, cPickle
import matplotlib.pyplot as plt
import numpy as np

def load_mnist():
    data_file = gzip.open('mnist.pkl.gz', 'rb')
    data = cPickle.load(data_file)
    data_file.close()
    return data


def plot_digits(data, num_cols, shape=(28,28)):
    num_digits = data.shape[0]
    num_rows = int(num_digits / num_cols)
    for i in range(num_digits):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.axis('off')
        plt.imshow(data[i].reshape(shape), interpolation='nearest', cmap='Greys')
    plt.show()


def get_digits(symbols,position_data,data):
	symbols_labeled = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
	for position in range (0,len(position_data)):
		symbols_labeled[position_data[position]] = symbols_labeled[position_data[position]] + [data[position]] #sort all symbols in the dict symbols_labeled
	#Het lukt me hier niet om de eerste 10 digits van elke list the plotten en ik heb geen idee waarom !?!
	return symbols_labeled


def plot_average(symbols_labeled):
    average_digit = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    for label in range (0,10):	
    	average_digit[label] = sum(symbols_labeled[label])
    plot_tmp = np.array([0]*7840)
    plot_tmp.shape = (10,784)
    for i in range (0,10):
    	plot_tmp[i] = average_digit[i]
    plot_tmp.shape = (10,784)
    plot_digits(plot_tmp,5)


if __name__ == "__main__":
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
    
    symbols_labeled = get_digits([0,1,2,3,4,5,6,7,8,9],t_train,x_train)
    plot_average(symbols_labeled)


    #plot_digits(x_train[:10], 5) #laat de eerset 10 plaatjes zien

    #print x_train.shape #waardes van de pixels
    #print t_train.shape #de goede antwoorden
    #print x_train[:10]
    #print t_train[:10]

	#inputnodes: 28**2 = 784
	#hiddenodes: ???
	#outputnodes: [0,0,0,0,0,0,0,0,0,0] = 10


    #TODO create dict with labels as keys and corresponding digits as values (dict1)

    #TODO print the first 10 members of every digit

    #TODO create dict with average digits (dict2)

    #TODO print the 10 average digits