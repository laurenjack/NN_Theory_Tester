import tensorflow as tf
import numpy as np
import math



def gaussian_mixture(a, means, A, batch_size, d):
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
    distance_squared = _weighted_distance_from_all_means(a, means, sigma, batch_size, d)
    exponent = 0.5 * (-distance_squared)
    pa_unnormed = tf.reduce_sum(tf.exp(exponent), axis=1)
    pa = 1.0 / (((2.0 * math.pi) ** d * sigma_determinant) ** 0.5 * number_of_means) * pa_unnormed
    return pa

def normal_seperate(a, mean, lam_inv, Q):
    diffs = a - mean
    eigen_space_diffs = tf.matmul(diffs, Q) * lam_inv
    exponents = -0.5 * eigen_space_diffs ** 2
    pa_unnormed = tf.exp(exponents)
    pa = 1.0 / (2.0 * math.pi) ** 0.5 * pa_unnormed
    return pa

def unscaled_eigen_prob(Q, lam_inv, a, centres, batch_size):
    """Return the eigen probabilities for any pdf based on the mean sum from Gaussian centres.
    """
    distances, true_exp = _eigen_distances_squared(Q, lam_inv, a, centres, batch_size)
    exponentials = tf.exp(-0.5 * distances)
    return tf.reduce_mean(exponentials, axis=1)

def eigen_probabilities(Q, lam_inv, a, centres, batch_size):
    """Return the eigen probabilities for any pdf based on the mean sum from Gaussian centres.
    """
    distances, true_exp = _eigen_distances_squared(Q, lam_inv, a, centres, batch_size)
    exponential = tf.exp(-0.5 * distances)
    mean_exp = tf.reduce_mean(exponential, axis=1)
    return 1.0 / (2.0 * math.pi) ** 0.5 * lam_inv * mean_exp


def product_of_kde(Q, lam_inv, a, centres, batch_size):
    eigen_probs = eigen_probabilities(Q, lam_inv, a, centres, batch_size)
    return tf.reduce_prod(eigen_probs, axis=1)
    # d, _ = _shape(Q)
    # d = float(d)
    # unscaled = unscaled_eigen_prob(Q, lam_inv, a, centres, batch_size)
    # return 1.0 / (2.0 * math.pi) ** (d / 2) * tf.reduce_prod(lam_inv ** d * unscaled, axis=1)




def chi_squared_distance_estimator(H_inverse, a, a_star, batch_size):
    """Model the distribution of the distance between points a and a_star using a chi_squared distribution

    Args:
        TODO(Jack)

    Returns:
        exponent - The chi-square exponent for the [batch_size, r] tensor a - a_star
        loss - The loss function, chi-square weighted sum of exponents
    """
    r, d = a_star.shape
    distance_squared = weighted_distance(H_inverse, a, a_star, batch_size)
    exponent = _chi_square_exponent(distance_squared)
    kernel = tf.exp(tf.reshape(exponent, [batch_size, r]))
    fa = kernel * tf.matrix_determinant(H_inverse) ** (1.0 / d)
    # exponent = tf.reshape(exponent, [batch_size, r]) + tf.log(tf.matrix_determinant(H_inverse)) * (1.0 / d)
    # loss = tf.nn.relu(exponent - self.min_chi_exponent)
    return exponent, fa

# def chi_square_kde_centered_exponent(H_inverse, a, a_star, batch_size, h):
#     r, d = a_star.shape
#     distance_squared = _weighted_distance(H_inverse, a, a_star, batch_size)
#     exponent = (_chi_square_exponent(distance_squared) - max_chi_exponent) / h
#     # exponent = self._chi_square_exponent(distance_squared / h)
#     kernel = tf.exp(exponent)
#     fa = tf.exp(max_chi_exponent) * tf.reduce_mean(tf.reshape(kernel, [batch_size, r]), axis=1) / 2.0 / h ** 0.5
#     return fa

def chi_squared_distribution(d, distance_squared):
    """
    Given a tensor of any non-zero shape, return the likelihood of X = distance_squared where X is a chi-squared
    distribution, for each element of distance_squared.
    """
    distance_squared = distance_squared
    exponent = _chi_square_exponent(d, distance_squared)
    return tf.exp(exponent)

def _eigen_distances_squared(Q, lam_inv, a, a_star, batch_size):
    r, d = _shape(a_star)
    difference = tf.reshape(a, [batch_size, 1, d]) - tf.reshape(a_star, [1, r, d])
    eigen_difference = tf.tensordot(difference, Q, axes=[[2], [0]]) * lam_inv
    true_exp = tf.matmul(eigen_difference, tf.transpose(eigen_difference, [0, 2, 1]))
    return eigen_difference ** 2.0, true_exp / d


def weighted_distance(H_inverse, a, a_star, batch_size):
    r, d = _shape(a_star)
    difference = tf.reshape(a, [batch_size, 1, d]) - tf.reshape(a_star, [1, r, d])
    weighted_difference = tf.reshape(tf.tensordot(difference, H_inverse, axes=[[2], [0]]),
                                  [batch_size, r, 1, d])
    distance_squared = tf.matmul(weighted_difference, tf.reshape(difference, [batch_size, r, d, 1]))
    return tf.reshape(distance_squared, [batch_size, r])

def _weighted_distance_from_all_means(a, means, sigma, batch_size, d):
    """For batch size examples in the tensor a of shape [batch_size, d], Find there weighted distance from all means
    as inversely weighted by sigma.
    """
    sigma_inverse = tf.matrix_inverse(sigma)
    number_of_means, difference = _difference_from_means(a, means, batch_size, d)
    difference = tf.reshape(difference, [batch_size, number_of_means, d, 1])
    distance_squared = tf.reshape(tf.tensordot(difference, sigma_inverse, axes=[[2], [0]]),
                                  [batch_size, number_of_means, 1, d])
    distance_squared = tf.matmul(distance_squared, difference)
    distance_squared = tf.reshape(distance_squared, [batch_size, number_of_means])
    return distance_squared

def _difference_from_means(a, means, batch_size, d):
    number_of_means = means.shape[0]
    a = tf.reshape(a, [batch_size, 1, d])
    means = means.reshape(1, number_of_means, d)
    difference = a - means
    return number_of_means, difference


def _chi_square_exponent(d, distance_squared):
    """Plug the tensor distance_squared into the chi-squared function.

    Args:
        distance_squared
    """
    exponent = (d / 2.0 - 1) * tf.log(distance_squared) - distance_squared / 2.0\
             - math.lgamma(d / 2.0) - d / 2.0 * tf.log(2.0)
    return exponent


def _shape(np_or_tf):
    shape = np_or_tf.shape
    if isinstance(np_or_tf, np.ndarray):
        return shape
    elif isinstance(np_or_tf, tf.Tensor) or isinstance(np_or_tf, tf.Variable):
        return [dim.value for dim in shape]
    raise ValueError("Expected an ndarray or tensor but got this object: {}".format(np_or_tf))

# d = 1000
# distance_sqaured = np.sum((np.random.randn(d) - np.random.randn(d)) ** 2.0)
# one_to_d_minus_one = np.arange(1, d / 2)
# foo = (d / 2.0 - 1) * np.log(distance_sqaured) - distance_sqaured / 2 - np.sum(np.log(one_to_d_minus_one)) - d / 2.0 * np.log(2)
# print foo
# # h = 1.0
# print np.exp(foo)


