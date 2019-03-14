import numpy as np
import tensorflow as tf
import math


class DataGenerator(object):
    """Class for generating data from known Gaussian distributions, for experimenting with distribution estimation.

    Instantiations of this class are specific to a randomly generated Gaussian in d dimensions. The matrix A is
    is generated by a random matrix of orthogonal eigen_vectors Q's and a uniform random choice of eigenvalues,
    in between the bounds specified by min_eigenvalue and min_eigenvalue. A = QLQt, where L is the diagonal matrix of
    eigenvalues.
    """
    def __init__(self, conf, random):
        self.random = random
        self.n = conf.n
        self.d = conf.d
        min_eigenvalue = conf.min_eigenvalue
        max_eigenvalue = conf.max_eigenvalue
        self.means = conf.means.astype(np.float32)
        self.number_of_means = self.means.shape[0]
        # Generate a standard deviation matrix A, by producing an eigen decomposition with a minimum limit for each
        # eigenvalue
        Q = _random_orthogonal_matrix(self.d)
        eigenvalues = random.uniform(min_eigenvalue, max_eigenvalue, self.d)
        upper_lambda = np.eye(self.d) * eigenvalues
        self.actual_A = np.matmul(Q, np.matmul(upper_lambda, Q.transpose()))
        self.sigma = np.matmul(self.actual_A.transpose(), self.actual_A)
        self.sigma_determinant = np.linalg.det(self.sigma)
        self.sigma_inverse = np.linalg.inv(self.sigma).astype(np.float32)

    def sample_gaussian_mixture(self):
        """Generates an [n, d] numpy array, where the n elements are drawn from a mixture of Gaussians in d dimensions

        Args:
            conf: The density_estimation configuration
            random: Service object encapsulating random behavior
            actual_A: The standard deviation matrix, the same one is applied across each Gaussian

        Return: A [n, d] numpy array of floating point numbers, drawn from the Gaussian mixture.
        """
        chosen_means = self.random.choice(self.means, self.n, replace=True)

        z = self.random.normal_numpy_array([self.n, self.d])
        return chosen_means + np.matmul(z, self.actual_A), self.actual_A

    def pdf(self, a, batch_size):
        """ Compute the PDF of the Gaussian mixture associated with this data generator, for batch_size points in a.

        Args:
            a: A [batch_size, d] tensor of observations in the distribution space (doesn't neccessarily have to be
            drawn from the Gaussian mixture).
            batch_size: A tensor, the number of examples in a.

        Return:
            The value of the pdf at for each point in a.
        """
        a = tf.reshape(a, [batch_size, 1, self.d])
        means = self.means.reshape(1, self.number_of_means, self.d)
        difference = a - means
        difference = tf.reshape(difference, [batch_size, self.number_of_means, self.d, 1])
        distance = tf.reshape(tf.tensordot(difference, self.sigma_inverse, axes=[[2], [0]]),
                   [batch_size, self.number_of_means, 1, self.d])
        distance = tf.matmul(distance, difference)
        distance = tf.reshape(distance, [batch_size, self.number_of_means])
        pa_unnormed = tf.reduce_sum(tf.exp(-0.5 * distance), axis=1)
        pa = 1.0 / (((2.0 * math.pi) ** self.d * self.sigma_determinant) ** 0.5 * self.number_of_means) * pa_unnormed
        return pa


def _random_orthogonal_matrix(d):
    H = np.eye(d)
    D = np.ones((d,))
    for n in range(1, d):
        x = np.random.normal(size=(d - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(d - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(d)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (d % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    Q = (D * H.T).T
    return Q