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
	#print len(position_data)
	for position in range (0,len(position_data)):
		symbols_labeled[position_data[position]] = symbols_labeled[position_data[position]] + [data[position]] #sort all symbols in the dict symbols_labeled
	#Het lukt me hier niet om de eerste 10 digits van elke list the plotten en ik heb geen idee waarom !?!
	#test = [0] * 10 #Er zijn blijkbaar niet van elk getal 5000 voorbeelden (!?!)
	#for i in range (0,10):
	#	print len(symbols_labeled[i])
	#	test[i] = len(symbols_labeled[i])
	#print sum(test)
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
    return average_digit

def get_diference(average_digits,data,position):
	diference = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
	for i in range (0,10):
		diference[i] = sum(abs(data[position] - average_digits[i]/len(symbols_labeled[i]))) / 784 #Er zijn blijkbaar niet van elk getal 5000 voorbeelden (!?!) dus daarom geef ik de waarde van symbols_labeled
		#print str(diference[i]*100) + "% between the average " + str(i)
	guess =  min(diference.items(), key=lambda x: x[1]) #het , key=lambda x: x[1]) gedeelte heb ik op internet gevonden, geen idee wat het betekent
	#print guess
	return guess[0]


class module_average:

	def __init__(self):   
		average_digits = plot_average(symbols_labeled)
		self.valid_average(x_valid,t_valid,average_digits)



	def valid_average(self,data,data_pos,average_digits):
	    answers_right = [0] * 10
	    answers_wrong = [0] * 10
	    for pos in range (0,len(data_pos)): #hij test meteen alles
	    	print "Testing " + str(pos+1) + "/" + str(len(data_pos)) + ". "
	    	guess = get_diference(average_digits,data,pos)
	    	#print "The right answer is: " + str(data_pos[pos])
	    	if guess == data_pos[pos]:
	    		answers_right[data_pos[pos]] = answers_right[data_pos[pos]] + 1.0
	    	else:
	    		answers_wrong[data_pos[pos]] = answers_wrong[data_pos[pos]] + 1.0
	    print "right: " + str(answers_right) + "!"
	    print "wrong: " + str(answers_wrong) + "!"
	    print sum(answers_right)/sum(answers_wrong)

if __name__ == "__main__":
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
    symbols_labeled = get_digits([0,1,2,3,4,5,6,7,8,9],t_train,x_train) 

    running_module = module_average()
    del running_module


    #plot_digits(x_train[:10], 5) #laat de eerset 10 plaatjes zien

    #print x_train.shape #waardes van de pixels
    #print t_train.shape #de goede antwoorden

	#inputnodes: 28**2 = 784
	#hiddenodes: ???
	#outputnodes: [0,0,0,0,0,0,0,0,0,0] = 10


    #DONE 1 create dict with labels as keys and corresponding digits as values (dict1)
    #DONE 3 create dict with average digits (dict2)
    #DONE 4 print the 10 average digits

    #TODO 2 print the first 10 members of every digit