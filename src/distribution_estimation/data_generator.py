import numpy as np


def generate_gaussian_mixture(conf, random):
    """Generates a 1D numpy array, where each element is drawn from a mixture of Gaussians specified be means.

    Args:
        conf: The density_estimation configuration
        random: Service object encapsulating random behavior

    Return: A 1D numpy array of floating point numbers, drawn from the Gaussian mixture
    """
    n = conf.n
    means = conf.means
    standard_deviation = conf.standard_deviation

    means = np.array(means)
    chosen_means = random.choice(means, n, replace=True)

    return chosen_means + standard_deviation * random.normal_numpy_array([n])