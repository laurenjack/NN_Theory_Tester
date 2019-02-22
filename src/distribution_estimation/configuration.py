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
        # The number of observations in the dataset.
        self.n = 10000
        # The number of examples for training at each step
        self.m = 1000
        # The number of reference examples (those part of the Kernel density estimate) for each training step
        self.r = 1000
        # The number of dimensions, for the random variable a
        self.d = 2
        # The initial value of R
        self.R_init = 0.5 * np.eye(2) # 0.5 * np.array([[1.0, 1.0 / 2.0 ** 0.5], [0.0, 1.0 / 2.0 ** 0.5]])
        # [float] - A list of means, one for each Gaussian in the actual distribution
        self.means = np.array([[-2.0, -2.0], [0.0, 2.0], [2.0, 2.0]])
        # float - The standard deviation matrix of the actual distribution, the same one is applied to each Gaussian
        self.actual_A = np.array([[1.0, -1.0 / 2.0 ** 0.5], [-1.0 / 2.0 ** 0.5, 1.0]])
        # The number of training epochs
        self.epochs = 20
        # The learning rate for R
        self.lr = 0.3
        # Floating point precision for tensorflow
        self.float_precision = tf.float32
        # Number of observations to be drawn when animating KDE versus actual_distribution
        self.number_of_animation_points = 300
        # Interval of time in milliseconds between steps in an animation
        self.animation_interval = 500
        # The following animation parameters apply only to the 2D case
        # The gap between the concentric ellipses drawn to represent the actual Gaussians, in standard deviations
        self.concentric_gap = 1.0
        # The max distance of the last concentric ellipse, in standard deviations.
        self.max_deviations = 4.0
        # For both the x and y axis, set the axis max and min (min will be the negative of this)
        self.axis_max_and_min = 8.0


        self._validate()


    def _validate(self):
        """Validate there aren't any inconsistencies in the configuration (just looks out for non-obvious ones).
        """

        if self.m + self.r > self.n:
            raise ValueError('There are {n} examples in the data set but more ({m} + {r}) are being drawn.'
                             .format(n=self.n, m=self.m, r=self.r))
