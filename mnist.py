import gzip, cPickle
import matplotlib.pyplot as plt

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

def get_symbol_position(symbols,position_data):
	dict1 = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
	n = 0
	while n < len(position_data):
		dict1[position_data[n]] = dict1[position_data[n]] + [n]
		n = n + 1
	print n
	n = 0
	while n < 10:
		list_print = dict1[n]
		print "The 10 first positions with " + str(n) + ": " + str(list_print[:10])
		n = n + 1
	return dict1


def plot_average(dict_positions):
    dict2 = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    k = 0
    while k < 10:
        list_temp = dict_positions[k]	
    	n = 0
    	while n < len(list_temp):
    		dict2[k] = dict2[k] + x_train[list_temp[n]]
    		#dict2[k] = dict2[k] / 10
    		#del list_temp[k]
    		n = n + 1
    	k = k + 1
    print dict2

    k = 0
    while k < 10:
    	print_temp = dict2[k]
    	print_temp.shape = (1,784)
    	print print_temp.shape
    	plot_digits(print_temp,1)
    	print len(list_temp)
    	k = k + 1


if __name__ == "__main__":
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
    
    dict1 = get_symbol_position([0,1,2,3,4,5,6,7,8,9],t_train)
    plot_average(dict1)


















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