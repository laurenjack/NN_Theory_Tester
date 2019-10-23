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
        self.fit_to_underlying_pdf = False
        # The number of observations in the dataset.
        self.n = 3600
        # The number of examples for training at each step
        self.m = 30
        # The number of reference examples (those part of the Kernel density estimate) for each training step
        self.r = 300
        # The number of dimensions, for the random variable a
        self.d = 100
        # The minimum and maximum eigenvalues of the underlying standard deviation matrix
        self.min_eigenvalue = 0.5
        self.max_eigenvalue = 2.0
        # The initial value of Q in f(a)
        Q, lam_inv = constant_creator.random_pd_Q_and_lam_inv(self.d, self.min_eigenvalue, self.max_eigenvalue)
        self.Q_init = Q + 0.001
        # The initial value of lam_inv in f(a)
        self.lam_inv_init = lam_inv #np.random.uniform(0.5, 2.0, size=[self.d]).astype(np.float32)
        # The weighting given to the objective function, 1 - k is given to the constraint. You must set 0 < k < 0.5
        self.k = 0.49
        # The degree to which the low variance eigen-bandwidths are scaled down.
        self.c = 0.4
        # [float] - A list of means, one for each Gaussian in the actual distribution
        self.means = np.zeros(shape=[1, self.d], dtype=np.float32)
        # The number of training epochs
        self.epochs = 200
        # The learning rate for R
        self.lr_init = 0.1 * self.d ** 0.5
        # The epochs when to apply a step wise decrese to the learning rate
        self.reduce_lr_epochs = [50, 100, 130, 160, 170, 180, 190]
        # The factor to scale the learning rate down by
        self.reduce_lr_factor = 0.8
        # Alternatively, this fixed A will override the random generation of A with an
        # pre-determined value, please leave as None if you don't want to do this.
        Q = np.eye(self.d)#constant_creator.random_orthogonal_matrix(self.d)
        lam = np.random.uniform(self.min_eigenvalue, self.max_eigenvalue, [self.d]) # np.eye(self.d, dtype=np.float32) *
        self.fixed_A = np.matmul(Q * lam, Q.transpose()) # np.random.uniform(0.5, 2.0, size=[self.d])
        self.show_variable_during_training = True