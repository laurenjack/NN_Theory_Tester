import os
import tarfile
import urllib

import numpy as np
from tensorflow.examples.tutorials import mnist


_MNIST_NUM_CLASS = 10
_CIFAR10_NUM_CLASS = 10
_CIFAR10_IMAGE_WIDTH = 32
_CIFAR10_VALIDATION_SET_PROPORTION = 0.1
_CIFAR10_BINARY_TAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
_CIFAR10_TAR_FILE_NAME = 'cifar-10-binary.tar.gz'
_TRAIN_FILE_NAMES = ['cifar-10-batches-bin/data_batch_{}.bin'.format(i + 1) for i in xrange(5)]
_TEST_FILE_NAME = 'cifar-10-batches-bin/test_batch.bin'


class Dataset:
    """Represents a data set for a neural network to be trained/tested/reported on.

    Note that all labels are scalars, (i.e. not one_hot). Therefore the Y attributes are vectors with n elements,
    where n is the number of examples in its respective data set. The X attributes can be in one of two forms.
    Either n * p where p is the total number of pixels in an image, or n * height * width * 3.

    Attributes:
        num_class: The number of class labels
        train: The Subset of examples for training
        validation: The Subset of examples for validation
        image_width: (optional) The width of the image inputs, if the inputs are structured as square images
    """

    def __init__(self, num_class, training_set, validation_set, image_width=None):
        self.num_class = num_class
        self.train = training_set
        self.validation = validation_set
        self.image_width = image_width


class Subset:
    """Dataset.Subset - Should only be used to represent the training/validation/test set of a data set.

    Attributes:
        x: The input examples, either an [n, d] matrix where d is the number of network inputs. Or an
        [n, image_width, image_width, 3] tensor (i.e. a set of images).
        y: An [n] vector of integers such that each element corresponds to a valid class, i.e.
        e is an element of Z: 0 <= e < num_class.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def n(self):
        """The number of examples in the subset"""
        return self.x.shape[0]


def load_mnist(random_targets=False):
    """Get the mnist data set, will download underlying files if they aren't locally present.

    Returns: A DataSet instance for MNIST. Where the images of each subset have the shape n * p
    e.g. (50000 images * 784 pixels).
    """
    # Load raw data.
    mnist_data = mnist.input_data.read_data_sets('MNIST_data', one_hot=False)

    # Create target vectors.
    x_train = mnist_data.train.images
    y_train = mnist_data.train.labels
    x_validation = mnist_data.validation.images
    y_validation = mnist_data.validation.labels

    # Create training and validation sets
    train_set = Subset(x_train, y_train)
    validation_set = Subset(x_validation, y_validation)
    return Dataset(_MNIST_NUM_CLASS, train_set, validation_set)


def load_cifar(data_dir, classes=None, random_targets=False):
    """Get the CIFAR 10 data set, will download and extract and normalise the data set if it doesn't exist in data_dir

    See https://www.cs.toronto.edu/~kriz/cifar.html If the data already exists it will simply be loaded from data_dir
    and normalized. When I say normalised, I really mean subtracting the per pixel mean and dividing by half the max
    range pixels can take on (128) This scales most pixels roughly between -1 and 1 (technically -2 < p < 2).

    Args:
        data_dir: The name of the directory that holds, or will hold, the unzipped CIFAR10 data files.
        classes: A binary tuple containing two integers from 0-9 each representing a CIFAR10 class. If this argument is
        provided, then only examples of these classes will be part of the data set (and all of such examples).

    Returns: A DataSet instance for CIFAR10. Where each individual images has the shape [IMAGE_WIDTH, IMAGE_WIDTH, 3].
    i.e. a subset of size n will have an x tensor of shape [n, IMAGE_WIDTH, IMAGE_WIDTH, 3]

    Raises:
        If classes is not none or not a binary tuple/list.
    """
    train_file_paths = [os.path.join(data_dir, file_name) for file_name in _TRAIN_FILE_NAMES]
    test_file_paths = [os.path.join(data_dir, _TEST_FILE_NAME)]
    _maybe_download(data_dir, train_file_paths + test_file_paths)

    # Load the non-test examples, split into training and validation
    x, y = _load_data(train_file_paths)
    # x, y = _shuffle(x, y)

    num_class = _CIFAR10_NUM_CLASS
    if classes:
        x, labels = _only_get_examples_of(classes, x, y)
        num_class = len(classes)

    validation_set_size = int(x.shape[0] * _CIFAR10_VALIDATION_SET_PROPORTION)
    x_train = x[validation_set_size:]
    y_train = y[validation_set_size:]
    x_validation = x[0:validation_set_size]
    y_validation = y[0:validation_set_size]

    # Use the pixel mean of the training set for normalisation
    pixel_mean = _compute_per_pixel_mean(x_train)
    x_train -= pixel_mean
    x_train /= 128.0
    x_validation -= pixel_mean
    x_validation /= 128.0

    train = Subset(x_train, y_train)
    validation = Subset(x_validation, y_validation)
    return Dataset(num_class, train, validation, _CIFAR10_IMAGE_WIDTH)


def _only_get_examples_of(classes, x, labels):
    """Only get examples which have their target class in classes.
    """
    if len(classes) != 2:
        raise ValueError('Should only specify None or a tuple of two classes, instead {} was specified'.format(classes))
    where_in_classes = np.logical_or(np.equal(labels, classes[0]), np.equal(labels, classes[1]))
    return x[where_in_classes], labels[where_in_classes]


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
    x = x.reshape(n, 3, _CIFAR10_IMAGE_WIDTH, _CIFAR10_IMAGE_WIDTH)
    x = x.transpose(0, 2, 3, 1)
    x = x.astype(dtype=np.float32)
    return x, y


def _compute_per_pixel_mean(x_train):
    n = x_train.shape[0]
    pixel_mean = np.mean(x_train[0:n], axis=0)
    return pixel_mean


def _shuffle(x, labels):
    n = x.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    return x[indices], labels[indices]