import tensorflow as tf
import numpy as np

import configuration
import data_generator
import kernel_density_estimator
import kde_trainer
import density_collector
import density_animator
from src import random_behavior

import matplotlib.pyplot as plt


def show_gradient():
    """
    Function for empirically validating the asymptotic properties of the gradient of the KDE training function.
    """
    conf = configuration.get_configuration()
    random = random_behavior.Random()

    # Initialise the distribution fitter
    kde = kernel_density_estimator.KernelDensityEstimator(conf)

    # Tensorflow setup
    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()



    r = conf.r
    fx_op, kernel_op = kde.pdf()

    total = 0.0
    for i in xrange(1):
        x, actual_A = data_generator.generate_gaussian_mixture(conf, random)
        a_star = x[0:r]
        a = x[r:]
        s = a.shape[0]

        fx, K = session.run([fx_op, kernel_op], feed_dict={kde.a: a, kde.a_star: a_star, kde.batch_size: s})
        fx = fx.reshape(s, 1)
        K = K.reshape(s, r)

        a_star = a_star.transpose()
        a_difference = a - a_star
        a_square = a_difference ** 2.0

        front_weight = K * a_difference / fx
        main_delta = np.mean(K * a_square, axis=1).reshape(s, 1) / fx - a_square
        complete_term = front_weight * main_delta
        reduced_over_a = np.mean(complete_term, axis=0)
        reduced_over_a_star = np.mean(reduced_over_a)
        full_term_before_reduction = np.mean(front_weight, axis=0) * reduced_over_a
        plt.scatter(a_star, np.mean(front_weight, axis=0))
        plt.scatter(a_star, reduced_over_a, color='r')
        plt.scatter(a_star, full_term_before_reduction, color='g')
        plt.show()
        print reduced_over_a_star
        total += reduced_over_a_star
    average = total / 1.0
    print '\nAverage: {}'.format(average)


if __name__ == '__main__':
    show_gradient()

# -0.036934608722087675
# -0.035532752567470856
# 0.07997541179226468