import tensorflow as tf
import numpy as np


conf = None  # singleton reference


def get_configuration():
    global conf
    if not conf:
        conf = Configuration()
    return conf


class Configuration:
    """Encapsulates the parameters regarding the structure, dataset, and training parameters of a Distribution
    Estimator.
    """

    def __init__(self):
        # The initial value of R
        self.R_init = np.array([[1.0]])
        # [float] - A list of means, one for each Gaussian in the real distribution
        self.means = [-4, 0.0, 7]
        # float - The standard deviation of each Gaussian
        self.standard_deviation = 1.0
        # The number of observations in the dataset.
        self.n = 10000
        # The number of training epochs
        self.epochs = 10
        # The number of examples for training at each step
        self.m = 1000
        # The number of reference examples (those part of the Kernel density estimate) for each training step
        self.r = 1000
        # The number of dimensions, for the random variable a
        self.d = 1
        # The learning rate for R
        self.lr = 0.3
        # Floating point precision for tensorflow
        self.float_precision = tf.float32
        # Number of observations to be drawn when animating KDE versus actual_distribution
        self.number_of_animation_points = 300
        # Interval of time in milliseconds between steps in an animation
        self.animation_interval = 100

        self._validate()


    def _validate(self):
        """Validate there aren't any inconsistencies in the configuration (just looks out for non-obvious ones).
        """

        if self.m + self.r > self.n:
            raise ValueError('There are {n} examples in the data set but more ({m} + {r}) are being drawn.'
                             .format(n=self.n, m=self.m, r=self.r))
