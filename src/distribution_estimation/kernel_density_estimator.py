import tensorflow as tf
import math


class KernelDensityEstimator(object):
    """Represents a Kernel Density Estimate with variable bandwidth h, over a set of reference samples a_star.
    """

    def __init__(self, conf):
        self.h = tf.Variable(conf.h_init, name='h', dtype=conf.float_precision)
        self.r = conf.r
        self.d = conf.d
        self.lr = conf.lr
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.a = tf.placeholder(dtype=conf.float_precision, shape=[None, self.d], name='a')
        self.a_star = tf.placeholder(dtype=conf.float_precision, shape=[self.r, self.d], name='a_star')

    def squared_weighted_mean_error(self):
        """This function returns a tensor to be minimised with respect to h (the bandwidth of the kernel function).

        Given a set of m observations: a, and a set of r reference observation a_star drawn from the same empirical
        distribution (which are used to give the Kernel Density Estimate), compute the squared weighted mean error,
        for each reference observation in a_star.
        """
        difference_H_basis, kernel, fa = self._compute_kernel()
        weighted_error = kernel * difference_H_basis / tf.reshape(fa, [self.batch_size, 1, 1]) # / self.h
        weighted_mean_error = tf.reduce_mean(weighted_error, axis=0)
        squared_weighted_mean_error = 0.5 * tf.reduce_mean(weighted_mean_error ** 2.0)
        train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(squared_weighted_mean_error)
        return train_op, squared_weighted_mean_error, self.h, tf.gradients(squared_weighted_mean_error, self.h)

    def pdf(self):
        """ For the set of values a, compute the relative likelihoods from the pdf of the kernel density estimator.

        Returns: A tensor, the relative likelihoods of each element of a.
        """
        _, _, fa = self._compute_kernel()
        det_H = tf.matrix_determinant(self.h ** 2.0)
        return 1.0 / ((2.0 * math.pi) ** self.d * det_H) ** 0.5 * fa

    def _compute_kernel(self):
        H = self.h ** 2.0
        # H_inverse = tf.reshape(tf.matrix_inverse(H), [1, self.d, self.d])
        H_inverse = tf.matrix_inverse(H)
        difference = tf.reshape(self.a, [self.batch_size, 1, self.d]) - tf.reshape(self.a_star, [1, self.r, self.d])
        difference_H_basis = tf.tensordot(difference, H_inverse ** 0.5, axes=[[2], [0]])
        distance_squared = tf.reshape(tf.tensordot(difference, H_inverse, axes=[[2], [0]]),
                                      [self.batch_size, self.r, 1, self.d])
        distance_squared = tf.matmul(distance_squared, tf.reshape(difference, [self.batch_size, self.r, self.d, 1]))
        # Drop one of the unnecessary 1 dimensions, leave the other for future broadcasting.
        distance_squared = tf.reshape(distance_squared, [self.batch_size, self.r, 1])
        exponent = -0.5 * distance_squared
        # No h as this cancels in the cost function
        kernel = tf.exp(exponent)
        fa = tf.reduce_mean(tf.reshape(kernel, [self.batch_size, self.r]), axis=1)
        return difference_H_basis, kernel, fa
