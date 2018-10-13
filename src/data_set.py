import os
import tarfile
import urllib

import numpy as np
from tensorflow.examples.tutorials import mnist


IMAGE_WIDTH = 32
CIFAR_10_BINARY_TAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
CIFAR_10_TAR_FILE_NAME = 'cifar-10-binary.tar.gz'
TRAIN_FILE_NAMES = ['cifar-10-batches-bin/data_batch_{}.bin'.format(i + 1) for i in xrange(5)]
TEST_FILE_NAME = 'cifar-10-batches-bin/test_batch.bin'


class DataSet:
    """Represents a data set for a neural network to be trained/tested/reported on.

    Note that all labels are scalars, (i.e. not one_hot).

    Attributes:
        X_train: training set images
        Y_train: training set labels
        X_val: validation set images
        Y_val: validation set labels
    """

    def __init__(self, X_train, Y_train, X_val, Y_val):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val


def load_mnist():
    """Get the mnist data set, will download underlying files if they aren't locally present.

    Returns: A DataSet instance for MNIST. Where each individual image is a vector (the rows of X_train or and X_val).
    i.e. X_train has the shape n * NUM_PIXELS
    """
    mnist_data = mnist.input_data.read_data_sets('MNIST_data', one_hot=False)
    train_set = mnist_data.train
    val_set = mnist_data.validation
    return DataSet(train_set.images, train_set.labels, val_set.images, val_set.labels)


def load_cifar(data_dir):
    """Get the CIFAR 10 data set, will download the underlying files if they aren't locally present.

    Args:
        data_dir: The directory that holds, or will hold, the unzipped CIFAR10 data folder.

    Returns: A DataSet instance for CIFAR10. Where each individual images has the shape IMAGE_WIDTH * IMAGE_WIDTH * 3.
    i.e. X_train has the shape n * IMAGE_WIDTH * IMAGE_WIDTH * 3
    """
    train_file_paths = [os.path.join(data_dir, file_name) for file_name in TRAIN_FILE_NAMES]
    test_file_paths = [os.path.join(data_dir, TEST_FILE_NAME)]
    _maybe_download(data_dir, train_file_paths + test_file_paths)
    x, y = _load_data(train_file_paths)
    x_test, y_test = _load_data(test_file_paths)  # TODO(Jack) change test to val
    pixel_mean = _compute_per_pixel_mean(x)
    x -= pixel_mean
    x /= 128.0
    x_test -= pixel_mean
    x_test /= 128.0
    return DataSet(x, y, x_test, y_test)


def _maybe_download(data_dir, cifar_file_paths):
    """Download the cifar-10 binary as linked at: https://www.cs.toronto.edu/~kriz/cifar.html if the data doesn't
    exist in the directory specified by the configuration.
    """
    all_exist = True
    for path in cifar_file_paths:
        all_exist = all_exist and os.path.exists(path)

    if not all_exist:
        print 'Downloading CIFAR10 data set from: '+CIFAR_10_BINARY_TAR_URL
        print 'Hang tight, this could take a few minutes'
        file_name, _ = urllib.urlretrieve(CIFAR_10_BINARY_TAR_URL, data_dir+CIFAR_10_TAR_FILE_NAME)
        tar = tarfile.open(file_name, 'r:gz')
        tar.extractall(data_dir)
        print 'Download and Extraction Complete'
        tar.close()


def _load_data(file_paths):
    xs = []
    for file_name in file_paths:
        x = np.fromfile(file_name, np.uint8)
        xs.append(x)
    x = np.concatenate(xs)
    n = len(file_paths) * 10000
    # Separate the labels
    x = x.reshape(n, -1)
    y = x[:, 0]
    x = x[:, 1:]
    # Reshape
    x = x.reshape(n, 3, IMAGE_WIDTH, IMAGE_WIDTH)
    x = x.transpose(0, 2, 3, 1)
    x = x.astype(dtype=np.float32)
    return x, y


def _compute_per_pixel_mean(x_train):
    pixel_mean = np.mean(x_train[0:50000], axis=0)
    return pixel_mean
