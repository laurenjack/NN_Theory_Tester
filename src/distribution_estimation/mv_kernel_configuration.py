import tensorflow as tf
import numpy as np

import constant_creator


conf = None  # singleton reference


def get_configuration():
    global conf
    if not conf:
        conf = MultivariateKernelConfiguration()
    return conf


class MultivariateKernelConfiguration:
    """Encapsulates the parameters regarding the structure, dataset, and training parameters of a Multivariate
    Kernel Desnity Estimator.
    """

    def __init__(self):
        # When true, trains on the actual pdf, i.e. minimise sum( (p(a) - f(a)) ** 2.0 ) directly. This is to isolate
        # the training parameters for tuning. When this is false, we use proper bandwith estimation without cheating
        self.fit_to_underlying_pdf = True
        # The number of observations in the dataset.
        self.n = 1200
        # The number of examples for training at each step
        self.m = 100
        # The number of reference examples (those part of the Kernel density estimate) for each training step
        self.r = 100
        # The number of dimensions, for the random variable a
        self.d = 2
        # The initial value of Q in f(a)
        self.Q_init = constant_creator.random_orthogonal_matrix(self.d) + 0.001
        # The initial value of lam_inv in f(a)
        self.lam_inv_init = np.array([0.5 / 0.42, 1.0 / 0.42], dtype=np.float32) #np.random.uniform(0.5, 2.0, size=[self.d]).astype(np.float32)
        # The weighting given to the objective function, 1 - k is given to the constraint. You must set 0 < k < 0.5
        self.k = 0.49
        # The degree to which the low variance eigen-bandwidths are scaled down.
        self.c = 0.4
        # [float] - A list of means, one for each Gaussian in the actual distribution
        self.means = np.zeros(shape=[1, self.d], dtype=np.float32)
        # The number of training epochs
        self.epochs = 100
        # The learning rate for R
        self.lr_init = 0.1
        # The epochs when to apply a step wise decrese to the learning rate
        self.reduce_lr_epochs = [40, 70]
        # The factor to scale the learning rate down by
        self.reduce_lr_factor = 0.1
        # The minimum and maximum eigenvalues of the underlying standard deviation matrix
        self.min_eigenvalue = 0.5
        self.max_eigenvalue = 2.0
        # Alternatively, this fixed A will override the random generation of A with an
        # pre-determined value, please leave as None if you don't want to do this.
        self.fixed_A = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.show_variable_during_training = True