import numpy as np


def generate_gaussian_mixture(conf, random):
    """Generates an [n, d], where the n elements are drawn from a mixture of Gaussians in d dimensions

    Args:
        conf: The density_estimation configuration
        random: Service object encapsulating random behavior
        actual_A: The standard deviation matrix, the same one is applied across each Gaussian

    Return: A 1D numpy array of floating point numbers, drawn from the Gaussian mixture.
    """
    n = conf.n
    d = conf.d
    means = conf.means
    actual_A = conf.actual_A

    chosen_means = random.choice(means, n, replace=True)

    z = random.normal_numpy_array([n, d])
    return chosen_means + np.matmul(z, actual_A)