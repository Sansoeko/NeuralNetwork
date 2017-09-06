# Log
(Change)log of the program and the research I'm doing.

### 28-06-2017
Created the empty github.com commit. 

### 01-07-2017
Ruben added the 'bare minimum' snake. To start learning python

### 04-07-2017
Started doing and did the todo's in snake.

### 28-08-2017
Ruben added mnist.pkl.gz and mnist.py including todo's
#####initial mnist.py:
```python
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
```

### 31-08-2017
I created an average_plotter. mnist.py can now take an average of all the data in x_train.

### 04-09-2017 
Optimized the code with Ruben (worked at least five hours straight to get everything 'neat'). Added classes and an 
accuracy test.

### 05-09-2017
Did some todo's and tweaked some minor things. Later Created the contrast model. Gave every model his own files. And 
sorted functions and Classes in their own files. I also updated the initial 'readme.md'. Later I fixed the log and Ruben
fixed some code and created 'todo.md'.
