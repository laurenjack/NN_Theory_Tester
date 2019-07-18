import tensorflow as tf
import numpy as np

from src.distribution_estimation import distribution_configuration
import repeated_estimation


def run(T):
    """Run the bandwidth estimation algorithm multiple times on a Gaussian mixture as specificied by the configuration.

    Over these T runs Plot the average h for each epoch. Show this average h against the optimal h.
    """
    # Create a configuration based on a standard normal distribution
    conf = distribution_configuration.get_configuration()
    conf.d = 1
    conf.fixed_A = np.array([[1.0]], dtype=np.float32)
    conf.means = np.array([[-1.0], [0.0], [1.5], [4.0]], dtype=np.float32)
    # Training Parameters
    conf.fit_to_underlying_pdf = False
    conf.show_variable_during_training = False
    conf.n = 10000
    conf.m = 100
    conf.r = 1000
    conf.R_init = 1.0 * np.eye(conf.d, dtype=np.float32)  # np.exp(-0.5) *
    conf.c = 0.2  # ** (1.0 / float(conf.d))
    conf.epochs = 100
    conf.lr_R = 0.05  # * (2 * math.pi * float(conf.d)) #** 0.5
    conf.reduce_lr_epochs = [24, 48, 72]
    # The factor to scale the learning rate down by
    conf.reduce_lr_factor = 0.3

    graph1 = tf.Graph()
    with graph1.as_default():
        print 'Fitting to f*(x) - i.e. p(x) unknown'
        mean_h, variance_h = repeated_estimation.run(conf, T)

    graph2 = tf.Graph()
    with graph2.as_default():
        conf.fit_to_underlying_pdf = True
        print 'Fitting directly to p(x)'
        mean_h_from_px, variance_h_from_px = repeated_estimation.run(conf, T)

    difference_of_means = mean_h - mean_h_from_px
    print '\nE(h) - E(h): {}'.format(difference_of_means)
    print 'Standard Error for difference {}'.format(((variance_h / T) + (variance_h_from_px / T)) ** 0.5)
    print ''


if __name__ == '__main__':
    run(30)