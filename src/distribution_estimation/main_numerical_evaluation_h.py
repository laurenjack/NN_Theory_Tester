import tensorflow as tf
import numpy as np

import distribution_configuration
import data_generator as dg
import kernel_density_estimator
import pdf_functions as pf
from src import random_behavior

import matplotlib.pyplot as plt


def show_loss_for(h_range):
    """ For a 1D Gaussian mixture, show the loss function for the bandwidth estimation algorithm at each h specified in
    h_range.

    Args:
        h_range: A numpy array of shape [?], specific the values of h you would like to compute a loss for.
    """
    conf = distribution_configuration.get_configuration()
    conf.d = 1
    random = random_behavior.Random()
    pdf_functions = pf.PdfFunctions(conf)
    data_generator = dg.GaussianMixture(conf, pdf_functions, random)

    x, _ = data_generator.sample(conf.n)

    # Initialise the distribution fitter
    kde = kernel_density_estimator.KernelDensityEstimator(conf, pdf_functions, data_generator)

    # Tensorflow setup
    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Create a placeholder to vary h
    A_inverse_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, 1], name = 'A_inverse_placeholder')
    r = conf.r
    s = conf.n - 2*r
    low_bias_A_inverse = tf.placeholder(dtype=tf.float32, shape=[conf.d, conf.d], name='low_bias_A_inverse')
    if conf.fit_to_underlying_pdf:
        loss_tensor = kde.loss(A_inverse_placeholder)[0]
    else:
        loss_tensor = kde.loss(A_inverse_placeholder, low_bias_A_inverse)[0]

    losses = []

    min_loss = 10.0 ** 10
    min_h = None
    number_of_h = h_range.shape[0]
    for i in xrange(number_of_h):
        h = h_range[i]
        print h
        A_inverse = np.array([[1.0 / h]])

        a_star1 = x[0:r]
        a_star2 = x[r:2*r]
        a = x[2*r:]

        feed_dict = {kde.a: a, kde.a_star1: a_star1, kde.a_star2: a_star2, kde.batch_size: s,
                     low_bias_A_inverse: np.array(5.0 * A_inverse), A_inverse_placeholder: A_inverse}
        loss = session.run(loss_tensor, feed_dict=feed_dict)
        if loss < min_loss:
            min_loss = loss
            min_h = h
        losses.append(loss)

    print '\nMinimum at {}'.format(min_h)
    losses = np.array(losses)
    plt.scatter(h_range, losses)
    plt.show()


if __name__ == '__main__':
    h_range = np.arange(0.05, 1.01, 0.01)
    show_loss_for(h_range)

# -0.036934608722087675
# -0.035532752567470856
# 0.07997541179226468