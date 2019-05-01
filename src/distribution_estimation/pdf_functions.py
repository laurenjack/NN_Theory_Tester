import tensorflow as tf
import numpy as np
import math


class PdfFunctions():
    """A class that represents various probability distributions, including Kernel Density estimators.
    """

    def __init__(self, conf):
        self.r = conf.r
        self.d = conf.d
        self.max_chi_exponent = self._chi_square_exponent(float(self.d - 2))
        self.min_chi_exponent = self._chi_square_exponent(float(4 * self.d))

    def chi_squared_distance_estimator(self, H_inverse, a, a_star, batch_size):
        """Model the distribution of the distance between points a and a_star using a chi_squared distribution

        Args:
            TODO(Jack)

        Returns:
            exponent - The chi-square exponent for the [batch_size, r] tensor a - a_star
            loss - The loss function, chi-square weighted sum of exponents
        """
        distance_squared = self._weighted_distance(H_inverse, a, a_star, batch_size)
        exponent = self._chi_square_exponent(distance_squared)
        kernel = tf.exp(tf.reshape(exponent, [batch_size, self.r]))
        fa = kernel * tf.matrix_determinant(H_inverse) ** (1.0 / self.d)
        # exponent = tf.reshape(exponent, [batch_size, self.r]) + tf.log(tf.matrix_determinant(H_inverse)) * (1.0 / self.d)
        # loss = tf.nn.relu(exponent - self.min_chi_exponent)
        return exponent, fa

    def chi_square_kde_centered_exponent(self, H_inverse, a, a_star, batch_size, h):
        distance_squared = self._weighted_distance(H_inverse, a, a_star, batch_size)
        exponent = (self._chi_square_exponent(distance_squared) - self.max_chi_exponent) / h
        # exponent = self._chi_square_exponent(distance_squared / h)
        kernel = tf.exp(exponent)
        fa = tf.exp(self.max_chi_exponent) * tf.reduce_mean(tf.reshape(kernel, [batch_size, self.r]), axis=1) / 2.0 / h ** 0.5
        return fa


    def normal(self, A_inverse, a, a_star, batch_size):
        """Compute f(a) for the [batch_size, d] set of points a, using the [r, d] set of reference points a_star, and
        the inverse bandwitdth matrix A_inverse, using normal kernels

        Args:
            A_inverse: A [d, d] tensor, the bandwidth of the kernel density estimate.
            a: The points in the batch to train on.
            a_star: The reference points which form the centres for the Kernel Density Estimate
            batch_size: A scalar tensor, the number of examples in the current batch

        Returns:
            A [batch_size] tensor. The relative likelihood f(a) for each element of a.
        """
        distance_squared = self._weighted_distance(A_inverse, a, a_star, batch_size)
        exponent = -0.5 * (distance_squared + self.d * tf.log(2 * math.pi))
        kernel = tf.exp(exponent)
        det_A_inverse = tf.matrix_determinant(A_inverse)
        fa_unscaled = tf.reduce_mean(tf.reshape(kernel, [self.batch_size, self.r]), axis=1)
        return det_A_inverse * fa_unscaled

    def chi_squared_distribution(self, distance_squared):
        """
        Given a tensor of any non-zero shape, return the likelihood of X = distance_squared where X is a chi-squared
        distribution, for each element of distance_squared.
        """
        distance_squared = distance_squared
        exponent = self._chi_square_exponent(distance_squared)
        return tf.exp(exponent)

    def _weighted_distance(self, H_inverse, a, a_star, batch_size):
        difference = tf.reshape(a, [batch_size, 1, self.d]) - tf.reshape(a_star, [1, self.r, self.d])
        distance_squared = tf.reshape(tf.tensordot(difference, H_inverse, axes=[[2], [0]]),
                                      [batch_size, self.r, 1, self.d])
        distance_squared = tf.matmul(distance_squared, tf.reshape(difference, [batch_size, self.r, self.d, 1]))
        # Drop one of the unnecessary 1 dimensions, leave the other for future broadcasting.
        return tf.reshape(distance_squared, [batch_size, self.r, 1])

    def _chi_square_exponent(self, distance_squared):
        """Plug the tensor distance_squared into the chi-squared function.

        Args:
            distance_squared
        """
        exponent = (self.d / 2.0 - 1) * tf.log(distance_squared) - distance_squared / 2.0\
                 - math.lgamma(self.d / 2.0) - self.d / 2.0 * tf.log(2.0)
        return exponent

# d = 1000
# distance_sqaured = np.sum((np.random.randn(d) - np.random.randn(d)) ** 2.0)
# one_to_d_minus_one = np.arange(1, d / 2)
# foo = (d / 2.0 - 1) * np.log(distance_sqaured) - distance_sqaured / 2 - np.sum(np.log(one_to_d_minus_one)) - d / 2.0 * np.log(2)
# print foo
# # h = 1.0
# print np.exp(foo)
