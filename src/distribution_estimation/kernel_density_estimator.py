import math

import tensorflow as tf
import numpy as np



class KernelDensityEstimator(object):
    """Represents a Kernel Density Estimate with variable bandwidth h, over a set of reference samples a_star.
    """

    def __init__(self, conf, data_generator):
        self.do_train = conf.do_train
        self.data_generator = data_generator
        self.R_inverse = tf.Variable(np.linalg.inv(conf.R_init), name='R', dtype=conf.float_precision)
        self.A_inverse = tf.matmul(self.R_inverse, tf.transpose(self.R_inverse))
        self.r = conf.r
        self.d = conf.d
        self.lr = conf.lr
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.a = tf.placeholder(dtype=conf.float_precision, shape=[None, self.d], name='a')
        self.a_star1 = tf.placeholder(dtype=conf.float_precision, shape=[self.r, self.d], name='a_star1')
        # self.a_star2 = tf.placeholder(dtype=conf.float_precision, shape=[self.r, self.d], name='a_star2')

    def total_loss(self, A_inverse):
        """This function returns a tensor to be minimised with respect to A_inverse (the inverse bandwidth matrix of the
        kernel function).

        A_inverse: For a Kernel Density estimate paramterised by the covariance matrix H = QLQt, A is the matrix
        A = Q(L^0.5)Qt. This parameter is the inverse of A and determines the bandwitdth of a multivariate kernel

        Take a set of m observations: a, and a set of r reference observation a_star drawn from the same empirical
        distribution of d dimensions. Compute the kernel density estimate compute the squared weighted mean error,
        for each reference observation in a_star. TODO(Jack) expand on this
        """
        fa = self.pdf(A_inverse)
        pa = self.data_generator.pdf(self.a, self.batch_size)
        loss = 0.5 * tf.reduce_mean((fa - pa) ** 2.0)
        train_op = None
        if self.do_train:
            train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(bias_loss)
        return train_op, loss

    def pdf(self, A_inverse):
        """Compute f(a) for the [batch_size, d] set of points a, using the [r, d] set of reference points and the
        inverse bandwitdth matrix A_inverse.

        Args:
            A_inverse: A [d, d] tensor, the bandwidth of the kernel density estimate.

        Returns:
            A [batch_size] tensor. The relative likelihood f(a) for each element of a.
        """
        H_inverse = tf.matmul(A_inverse, tf.transpose(A_inverse))
        difference = tf.reshape(self.a, [self.batch_size, 1, self.d]) - tf.reshape(self.a_star1, [1, self.r, self.d])
        distance_squared = tf.reshape(tf.tensordot(difference, H_inverse, axes=[[2], [0]]),
                                      [self.batch_size, self.r, 1, self.d])
        distance_squared = tf.matmul(distance_squared, tf.reshape(difference, [self.batch_size, self.r, self.d, 1]))
        # Drop one of the unnecessary 1 dimensions, leave the other for future broadcasting.
        distance_squared = tf.reshape(distance_squared, [self.batch_size, self.r, 1])
        exponent = -0.5 * distance_squared
        kernel = tf.exp(exponent)
        det_A_inverse = tf.matrix_determinant(A_inverse)
        fa_unscaled = tf.reduce_mean(tf.reshape(kernel, [self.batch_size, self.r]), axis=1)
        return det_A_inverse / (2.0 * math.pi) ** (self.d * 0.5) * fa_unscaled
