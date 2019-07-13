import tensorflow as tf
import math


class PdfFunctions(object):
    """A class that represents various probability distributions.
    """

    def gaussian_mixture(self, a, means, A, batch_size, d):
        """ Compute the PDF of the Gaussian mixture associated with this data generator, for batch_size points in a.

        Args:
            a: A [batch_size, d] tensor of observations in the distribution space (doesn't neccessarily have to be
            drawn from the Gaussian mixture).
            A: The matrix Sigma=AtA, such that Sigma is the covariance matrix
            batch_size: A tensor, the number of examples in a.
            d: The number of dimension of a.

        Return:
            The value of the pdf at for each point in a.
        """
        number_of_means = means.shape[0]
        sigma = tf.matmul(tf.transpose(A), A)
        sigma_determinant = tf.matrix_determinant(sigma)
        distance_squared = self._weighted_distance_from_all_means(a, means, sigma, batch_size, d)
        exponent = 0.5 * (-distance_squared)
        pa_unnormed = tf.reduce_sum(tf.exp(exponent), axis=1)
        pa = 1.0 / (((2.0 * math.pi) ** d * sigma_determinant) ** 0.5 * number_of_means) * pa_unnormed
        return pa

    def chi_squared_distance_estimator(self, H_inverse, a, a_star, batch_size):
        """Model the distribution of the distance between points a and a_star using a chi_squared distribution

        Args:
            TODO(Jack)

        Returns:
            exponent - The chi-square exponent for the [batch_size, r] tensor a - a_star
            loss - The loss function, chi-square weighted sum of exponents
        """
        r, d = a_star.shape
        distance_squared = self._weighted_distance(H_inverse, a, a_star, batch_size)
        exponent = self._chi_square_exponent(distance_squared)
        kernel = tf.exp(tf.reshape(exponent, [batch_size, r]))
        fa = kernel * tf.matrix_determinant(H_inverse) ** (1.0 / d)
        # exponent = tf.reshape(exponent, [batch_size, r]) + tf.log(tf.matrix_determinant(H_inverse)) * (1.0 / d)
        # loss = tf.nn.relu(exponent - self.min_chi_exponent)
        return exponent, fa

    def chi_square_kde_centered_exponent(self, H_inverse, a, a_star, batch_size, h):
        r, d = a_star.shape
        distance_squared = self._weighted_distance(H_inverse, a, a_star, batch_size)
        exponent = (self._chi_square_exponent(distance_squared) - self.max_chi_exponent) / h
        # exponent = self._chi_square_exponent(distance_squared / h)
        kernel = tf.exp(exponent)
        fa = tf.exp(self.max_chi_exponent) * tf.reduce_mean(tf.reshape(kernel, [batch_size, r]), axis=1) / 2.0 / h ** 0.5
        return fa

    def chi_squared_distribution(self, d, distance_squared):
        """
        Given a tensor of any non-zero shape, return the likelihood of X = distance_squared where X is a chi-squared
        distribution, for each element of distance_squared.
        """
        distance_squared = distance_squared
        exponent = self._chi_square_exponent(d, distance_squared)
        return tf.exp(exponent)

    def _weighted_distance(self, H_inverse, a, a_star, batch_size):
        r, d = a_star.shape
        difference = tf.reshape(a, [batch_size, 1, d]) - tf.reshape(a_star, [1, r, d])
        distance_squared = tf.reshape(tf.tensordot(difference, H_inverse, axes=[[2], [0]]),
                                      [batch_size, r, 1, d])
        distance_squared = tf.matmul(distance_squared, tf.reshape(difference, [batch_size, r, d, 1]))
        return tf.reshape(distance_squared, [batch_size, r])

    def _weighted_distance_from_all_means(self, a, means, sigma, batch_size, d):
        """For batch size examples in the tensor a of shape [batch_size, d], Find there weighted distance from all means
        as inversely weighted by sigma.
        """
        number_of_means = means.shape[0]
        sigma_inverse = tf.matrix_inverse(sigma)
        a = tf.reshape(a, [batch_size, 1, d])
        means = means.reshape(1, number_of_means, d)
        difference = a - means
        difference = tf.reshape(difference, [batch_size, number_of_means, d, 1])
        distance_squared = tf.reshape(tf.tensordot(difference, sigma_inverse, axes=[[2], [0]]),
                                      [batch_size, number_of_means, 1, d])
        distance_squared = tf.matmul(distance_squared, difference)
        distance_squared = tf.reshape(distance_squared, [batch_size, number_of_means])
        return distance_squared

    def _chi_square_exponent(self, d, distance_squared):
        """Plug the tensor distance_squared into the chi-squared function.

        Args:
            distance_squared
        """
        exponent = (d / 2.0 - 1) * tf.log(distance_squared) - distance_squared / 2.0\
                 - math.lgamma(d / 2.0) - d / 2.0 * tf.log(2.0)
        return exponent

# d = 1000
# distance_sqaured = np.sum((np.random.randn(d) - np.random.randn(d)) ** 2.0)
# one_to_d_minus_one = np.arange(1, d / 2)
# foo = (d / 2.0 - 1) * np.log(distance_sqaured) - distance_sqaured / 2 - np.sum(np.log(one_to_d_minus_one)) - d / 2.0 * np.log(2)
# print foo
# # h = 1.0
# print np.exp(foo)
