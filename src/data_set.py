from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import configuration
conf = configuration.get_configuration()

class DataSet:
    """Class represents a data set for a neural network to be trained/tested/validated on"""

    def __init__(self, X_train, Y_train, X_val, Y_val):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val

def load_mnist():
    """Returns the mnist data set"""
    # Load data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    train_set = mnist.train
    val_set = mnist.validation
    return DataSet(train_set.images, train_set.labels, val_set.images, val_set.labels)


def load_cifar():
    # TODO change test to val
    x, y = _load_data(conf.train_files)
    x_test, y_test = _load_data([conf.test_file])
    pixel_mean = _compute_per_pixel_mean(x)
    x -= pixel_mean
    x /= 128.0
    x_test -= pixel_mean
    x_test /= 128.0
    return DataSet(x, y, x_test, y_test)


def _load_data(filenames):
    xs = []
    for file_name in filenames:
        x = np.fromfile(file_name, np.uint8)
        xs.append(x)
    x = np.concatenate(xs)
    n = len(filenames) * 10000
    # Seperate labels
    x = x.reshape(n, -1)
    y = x[:, 0]
    x = x[:, 1:]
    # Reshape
    x = x.reshape(n, 3, conf.image_width, conf.image_width)
    x = x.transpose(0, 2, 3, 1)
    x = x.astype(dtype=np.float32)
    return x, y

def _compute_per_pixel_mean(x_train):
    pixel_mean = np.mean(x_train[0:50000], axis=0)
    return pixel_mean