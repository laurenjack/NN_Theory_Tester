import tensorflow as tf
import numpy as np

import configuration
import data_generator as dg
import kernel_density_estimator
import trainer
import density_collector
import density_animator
from src import random_behavior

import matplotlib.pyplot as plt


def show_variance_function():
    conf = configuration.get_configuration()
    random = random_behavior.Random()
    data_generator = dg.DataGenerator(conf, random)

    x, _ = data_generator.sample_gaussian_mixture()

    # Initialise the distribution fitter
    kde = kernel_density_estimator.KernelDensityEstimator(conf, data_generator)

    # Tensorflow setup
    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Create a placeholder to vary h
    A_inverse_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, 1], name = 'A_inverse_placeholder')
    r = conf.r
    s = conf.n - 2*r
    low_bias_A_inverse = tf.placeholder(dtype=tf.float32, shape=[conf.d, conf.d], name='low_bias_A_inverse')
    if conf.fit_to_underlying_pdf:
        loss_tensor = kde.loss(A_inverse_placeholder)
    else:
        loss_tensor = kde.loss(A_inverse_placeholder, low_bias_A_inverse)

    hs = np.arange(0.05, 1.01, 0.01)
    losses = []

    number_of_h = hs.shape[0]
    for i in xrange(number_of_h):
        h = hs[i]
        print h
        A_inverse = np.array([[1.0 / h]])

        a_star1 = x[0:r]
        a_star2 = x[r:2*r]
        a = x[2*r:]

        feed_dict = {kde.a: a, kde.a_star1: a_star1, kde.a_star2: a_star2, kde.batch_size: s,
                     low_bias_A_inverse: np.array(5.0 * A_inverse), A_inverse_placeholder: A_inverse}
        loss = session.run(loss_tensor, feed_dict=feed_dict)
        losses.append(loss)

    losses = np.array(losses)
    plt.scatter(hs, losses)
    plt.show()



if __name__ == '__main__':
    #show_gradient_bias()
    show_variance_function()

# -0.036934608722087675
# -0.035532752567470856
# 0.07997541179226468