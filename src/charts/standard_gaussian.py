import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from src.distribution_estimation import distribution_configuration
from src.distribution_estimation import pdf_functions as pf
from  src.distribution_estimation import data_generator as dg
from src.distribution_estimation import kernel_density_estimator
from src import random_behavior


def run(r, h, fa_title, single_kernel=False):
    """Plot a standard Gaussian distribution.
    """
    # Create a configuration based on a standard normal distribution
    conf = distribution_configuration.get_configuration()
    conf.r = r
    conf.d = 1
    # No variability in A, just set it to 1 for a standard normal Gaussian
    conf.fixed_A = np.array([[1.0]], dtype=np.float32)
    conf.means = np.array([[0.0]], dtype=np.float32)

    # Construct the required services
    random = random_behavior.Random()
    gaussian_mixture = dg.GaussianMixture(conf, random)
    kde = kernel_density_estimator.KernelDensityEstimator(conf, gaussian_mixture)

    # Create some points to display p(a)
    a = np.arange(-5, 5.01, 0.01).astype(np.float32)
    batch_size = a.shape[0]
    a = a.reshape(batch_size, 1)
    # Draw r reference points from the gaussian
    a_star = gaussian_mixture.sample(r)
    # Check if the user wants to display a single kernel instead
    if single_kernel and r == 1:
        a_star = np.array([[-0.5]], dtype=np.float32)

    # Compute p(a)
    pa_tensor = gaussian_mixture.pdf(a, batch_size)
    session = tf.Session()
    pa = session.run(pa_tensor)
    # Compute p(-1) for the bias - variance example
    p_neg_1_tensor = gaussian_mixture.pdf([[-0.5]], 1)
    p_neg_1 = session.run(p_neg_1_tensor)
    # Compute f(a)
    A_inverse = np.array([[1.0 / h]], dtype=np.float32)
    fa_tensor = kde.pdf(A_inverse, a_star)
    fa = session.run(fa_tensor, feed_dict={kde.a: a, kde.batch_size: batch_size})

    # Graph p(a) for every a
    _plot_normal_and_kde(a, pa, p_neg_1, 1)
    # Same again but with f(a) too
    _plot_normal_and_kde(a, pa, p_neg_1, 2, fa, fa_title)
    plt.show()


def _plot_normal_and_kde(a, pa, p_neg_1, fig_number, fa=None, fa_title=None):
    plt.figure(fig_number)
    title = 'Standard Normal Distribution'
    plt.plot(a, pa)
    plt.scatter([-0.5], p_neg_1, color='k')
    if fa is not None and fa_title is not None:
        title += ' - '+fa_title
        plt.plot(a, fa, color='r')
        ax = plt.gca()
        ax.set_ylim(0, 2)
    plt.title(title)


if __name__ == '__main__':
    r = 1
    # Optimal h - Silverman's rule of thumb is the analytical minimum of Integrated mean square error for a Gaussian.
    h = (4.0 / (3.0 * r)) ** 0.2
    print h
    title = 'KDE with Optimal h'
    # High Bias h
    h = 0.8
    title = 'KDE with h={} - High Bias'.format(h)
    # High Variance h
    h = 0.08
    title = 'KDE with h={} - High Variance'.format(h)

    title = 'Single Kernel with h={}'.format(h)

    run(r, h, title, True)