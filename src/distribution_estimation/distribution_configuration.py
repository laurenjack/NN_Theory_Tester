import tensorflow as tf
import numpy as np
import math


conf = None  # singleton reference


def get_configuration():
    global conf
    if not conf:
        conf = DensityConfiguration()
    return conf


class DensityConfiguration: #TODO(Jack) update documentation to reflect new configuration
    """Encapsulates the parameters regarding the structure, dataset, and training parameters of a Distribution
    Estimator.
    """

    def __init__(self):
        # When true, trains on the actual pdf, i.e. minimise sum( (p(a) - f(a)) ** 2.0 ) directly. This is to isolate
        # the training parameters for tuning. When this is false, we use proper bandwith estimation without cheating
        self.fit_to_underlying_pdf = True
        # The number of observations in the dataset.
        self.n = 10000
        # The number of examples for training at each step
        self.m = 1000
        # The number of reference examples (those part of the Kernel density estimate) for each training step
        self.r = 1000
        # The number of dimensions, for the random variable a
        self.d = 1
        # The initial value of R
        self.R_init = 1.0 * np.eye(self.d) # np.exp(-0.5) *
        # The degree to which the bandwidth matrix of the estimate is scaled.
        self.c = 0.2 #** (1.0 / float(self.d))
        # [float] - A list of means, one for each Gaussian in the actual distribution
        self.means = np.zeros((1, self.d))  # np.concatenate([-1.0 * np.ones((1, self.d)), 1.0 * np.ones((1, self.d))])
        # The number of training epochs
        self.epochs = 5
        # The learning rate for R
        self.lr_R = 0.006 #* (2 * math.pi * float(self.d)) #** 0.5
        # Floating point precision for tensorflow
        self.float_precision = tf.float32
        # The minimum and maximum eigenvalues of the underlying standard deviation matrix
        self.min_eigenvalue = 1.0
        self.max_eigenvalue = 1.0
        # Alternatively, this fixed A will override the random generation of A with an
        # pre-determined value, please leave as None if you don't want to do this.
        self.fixed_A = None
        # Show A each after each training batch
        self.show_variable_during_training = True
        # Number of observations to be drawn when animating KDE versus actual_distribution
        self.number_of_animation_points = 300
        # Interval of time in milliseconds between steps in an animation
        self.animation_interval = 100
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

    def _gen_positive_definate_matrix(self):
        a = np.random.randn(self.d, self.d)
        magnitude = np.sum(a ** 2.0, axis=1) ** 0.5
        a = a / magnitude.reshape(self.d, 1)
        return np.matmul(a.transpose(), a)

