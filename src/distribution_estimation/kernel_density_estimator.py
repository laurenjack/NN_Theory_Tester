import math

import tensorflow as tf
import numpy as np



class KernelDensityEstimator(object):
    """Represents a Kernel Density Estimate with variable bandwidth h, over a set of reference samples a_star.
    """

    def __init__(self, conf):
        self.R_inverse = tf.Variable(np.linalg.inv(conf.R_init), name='R', dtype=conf.float_precision)
        self.A_inverse = tf.matmul(self.R_inverse, tf.transpose(self.R_inverse))
        self.r = conf.r
        self.d = conf.d
        self.lr = conf.lr
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.a = tf.placeholder(dtype=conf.float_precision, shape=[None, self.d], name='a')
        self.a_star1 = tf.placeholder(dtype=conf.float_precision, shape=[self.r, self.d], name='a_star1')
        # self.a_star2 = tf.placeholder(dtype=conf.float_precision, shape=[self.r, self.d], name='a_star2')

    def total_loss(self):
        """This function returns a tensor to be minimised with respect to h (the bandwidth of the kernel function).

        Given a set of m observations: a, and a set of r reference observation a_star drawn from the same empirical
        distribution (which are used to give the Kernel Density Estimate), compute the squared weighted mean error,
        for each reference observation in a_star.
        """
        return self._total_loss(self.A_inverse)


    def _total_loss(self, A_inverse):
        difference, kernel, fa1 = self._compute_kernel(A_inverse)
        # _, _, fa2 = self._compute_kernel(A_inverse, self.a_star1)

        weighted_error = kernel * difference / (tf.reshape(fa1, [self.batch_size, 1, 1]) + 10.0 ** -30)
        weighted_mean_error = tf.reduce_mean(weighted_error, axis=0)
        bias_loss = 0.5 * tf.reduce_mean(weighted_mean_error ** 2.0)

        # train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(bias_loss)
        return None, bias_loss, A_inverse, \
               tf.gradients(bias_loss, self.R_inverse), tf.reduce_min(self.pdf()[0])


    def pdf(self):
        """ For the set of values a, compute the relative likelihoods from the pdf of the kernel density estimator.

        Returns: A tensor, the relative likelihoods of each element of a.
        """
        return self._pdf(self.A_inverse)

    def _pdf(self, A_inverse):
        _, kernel, fa = self._compute_kernel(A_inverse)
        det_A_inverse = tf.matrix_determinant(A_inverse)
        return det_A_inverse / (2.0 * math.pi) ** (self.d * 0.5) * fa, kernel

    def _compute_kernel(self, A_inverse):
        H_inverse = tf.matmul(A_inverse, tf.transpose(A_inverse))
        difference = tf.reshape(self.a, [self.batch_size, 1, self.d]) - tf.reshape(self.a_star1, [1, self.r, self.d])
        # difference_A_basis = tf.tensordot(difference, A_inverse, axes=[[2], [0]])
        distance_squared = tf.reshape(tf.tensordot(difference, H_inverse, axes=[[2], [0]]),
                                      [self.batch_size, self.r, 1, self.d])
        distance_squared = tf.matmul(distance_squared, tf.reshape(difference, [self.batch_size, self.r, self.d, 1]))
        # Drop one of the unnecessary 1 dimensions, leave the other for future broadcasting.
        distance_squared = tf.reshape(distance_squared, [self.batch_size, self.r, 1])
        exponent = -0.5 * distance_squared
        # No h as this cancels in the cost function
        kernel = tf.exp(exponent)
        fa = tf.reduce_mean(tf.reshape(kernel, [self.batch_size, self.r]), axis=1)
        #return difference_A_basis, kernel, fa
        return difference, kernel, fa
