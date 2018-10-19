import os
import tarfile
import urllib

import numpy as np
from tensorflow.examples.tutorials import mnist


_IMAGE_WIDTH = 32
_CIFAR10_VALIDATION_SET_SIZE = 5000
_CIFAR10_BINARY_TAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
_CIFAR10_TAR_FILE_NAME = 'cifar-10-binary.tar.gz'
_TRAIN_FILE_NAMES = ['cifar-10-batches-bin/data_batch_{}.bin'.format(i + 1) for i in xrange(5)]
_TEST_FILE_NAME = 'cifar-10-batches-bin/test_batch.bin'


class DataSet:
    """Represents a data set for a neural network to be trained/tested/reported on.

    Note that all labels are scalars, (i.e. not one_hot). Therefore the Y attributes are vectors with n elements,
    where n is the number of examples in its respective data set. The X attributes can be in one of two forms.
    Either n * p where p is the total number of pixels in an image, or n * height * width * 3.

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

    @property
    def n_train(self):
        """The number of examples in the training set."""
        return self.X_train.shape[0]

    @property
    def n_val(self):
        """The number of examples in the validation set."""
        return self.X_val.shape[0]


def load_mnist():
    """Get the mnist data set, will download underlying files if they aren't locally present.

    Returns: A DataSet instance for MNIST. Where each individual image is a vector (the rows of X_train or and X_val).
    i.e. X_train has the shape n * p (50000 images * 784 pixels)
    """
    mnist_data = mnist.input_data.read_data_sets('MNIST_data', one_hot=False)
    train_set = mnist_data.train
    val_set = mnist_data.validation
    return DataSet(train_set.images, train_set.labels, val_set.images, val_set.labels)


def load_cifar(data_dir):
    """Get the CIFAR 10 data set, will download and extract and normalise the data set if it doesn't exist in data_dir

    See https://www.cs.toronto.edu/~kriz/cifar.html If the data already exists it will simply be loaded from data_dir
    and normalized. When I say normalised, I really mean subtracting the per pixel mean and dividing by half the max
    range pixels can take on (128) This scales most pixels roughly between -1 and 1 (technically -2 < p < 2).

    Args:
        data_dir: The directory that holds, or will hold, the unzipped CIFAR10 data files. This is the directory the
        user specified as a program argument.

    Returns: A DataSet instance for CIFAR10. Where each individual images has the shape IMAGE_WIDTH * IMAGE_WIDTH * 3.
    i.e. X_train has the shape n * IMAGE_WIDTH * IMAGE_WIDTH * 3.
    """
    train_file_paths = [os.path.join(data_dir, file_name) for file_name in _TRAIN_FILE_NAMES]
    test_file_paths = [os.path.join(data_dir, _TEST_FILE_NAME)]
    _maybe_download(data_dir, train_file_paths + test_file_paths)

    # Load the non-test examples, split into training and validation
    x, y = _load_data(train_file_paths)
    x, y = _shuffle(x, y)
    x_train = x[_CIFAR10_VALIDATION_SET_SIZE:]
    y_train = y[_CIFAR10_VALIDATION_SET_SIZE:]
    x_val = x[0:_CIFAR10_VALIDATION_SET_SIZE]
    y_val = y[0:_CIFAR10_VALIDATION_SET_SIZE]

    # Use the pixel mean of the training set for normalisation
    pixel_mean = _compute_per_pixel_mean(x_train)
    x_train -= pixel_mean
    x_train /= 128.0
    x_val -= pixel_mean
    x_val /= 128.0
    return DataSet(x_train, y_train, x_val, y_val)


def _maybe_download(data_dir, cifar_file_paths):
    """Download the cifar-10 binary as linked at: https://www.cs.toronto.edu/~kriz/cifar.html if the data doesn't
    exist in the directory specified by the configuration.
    """
    all_exist = True
    for path in cifar_file_paths:
        all_exist = all_exist and os.path.exists(path)

    if not all_exist:
        print 'Downloading CIFAR10 data set from: ' + _CIFAR10_BINARY_TAR_URL
        print 'Hang tight, this could take a few minutes'
        file_name, _ = urllib.urlretrieve(_CIFAR10_BINARY_TAR_URL, data_dir + _CIFAR10_TAR_FILE_NAME)
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
    x = x.reshape(n, 3, _IMAGE_WIDTH, _IMAGE_WIDTH)
    x = x.transpose(0, 2, 3, 1)
    x = x.astype(dtype=np.float32)
    return x, y


def _compute_per_pixel_mean(x_train):
    n = x_train.shape[0]
    pixel_mean = np.mean(x_train[0:n], axis=0)
    return pixel_mean


def _shuffle(x, y):
    n = x.shape[0]
    inds = np.arange(n)
    np.random.shuffle(inds)
    return x[inds], y[inds]