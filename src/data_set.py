from tensorflow.examples.tutorials.mnist import input_data


def load_mnist():
    """Returns the mnist data set"""
    # Load data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    train_set = mnist.train
    val_set = mnist.validation
    return DataSet(train_set.images, train_set.labels, val_set.images, val_set.labels)

class DataSet:
    """Class represents a data set for a neural network to be trained/tested/validated on"""

    def __init__(self, X_train, Y_train, X_val, Y_val):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val