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


def show_gradient_bias():
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


def show_variance_function():
    conf = configuration.get_configuration()
    random = random_behavior.Random()

    # Initialise the distribution fitter
    kde = kernel_density_estimator.KernelDensityEstimator(conf)

    # Tensorflow setup
    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Create a placeholder to very h over time
    A_inverse_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, 1], name = 'A_inverse_placeholder')
    r = conf.r
    s = conf.n - 3*r
    fx_op, kernel_op = kde._pdf(A_inverse_placeholder)
    _, loss_bias_op, _, _, _ = kde._total_loss(A_inverse_placeholder)

    x, actual_A = data_generator.generate_gaussian_mixture(conf, random)

    hs = np.arange(0.05, 1.01, 0.01)
    loss_vars = []
    loss_biases = []

    number_of_h = hs.shape[0]
    for i in xrange(number_of_h):
        h = hs[i]
        print h
        A_inverse = np.array([[1.0 / h]])

        a_star1 = x[0:r]
        a_star2 = x[r: 2*r]
        a_star3 = x[2*r: 3*r]
        a = x[3*r:]

        fx1 = session.run(fx_op, feed_dict={kde.a: a, kde.a_star1: a_star1, kde.batch_size: s,
                                                            A_inverse_placeholder: A_inverse})
        fx2 = session.run(fx_op, feed_dict={kde.a: a, kde.a_star1: a_star2, kde.batch_size: s,
                                                            A_inverse_placeholder: A_inverse})
        loss_var = np.mean(((fx1 - fx2)) ** 2.0)
        loss_vars.append(loss_var)

        # fx3, K3 = session.run([fx_op, kernel_op], feed_dict={kde.a: a, kde.a_star: a_star3, kde.batch_size: s,
        #                                                    A_inverse_placeholder: A_inverse})
        loss_bias = session.run(loss_bias_op, feed_dict={kde.a: a, kde.a_star1: a_star3, kde.batch_size: s,
                                                             A_inverse_placeholder: A_inverse})
        loss_biases.append(loss_bias)
        # a_star3 = a_star3.transpose()
        # a_difference = a - a_star3
        # fx3 = fx3.reshape(s, 1)
        # mean_diff = np.mean(K3 * a_difference / fx3, axis=1)
        # loss_bias = np.mean(mean_diff ** 2.0)

    loss_vars = np.array(loss_vars)
    loss_biases = np.array(loss_biases)
    plt.scatter(hs, loss_vars)
    plt.scatter(hs, loss_biases, color='r')
    plt.scatter(hs, loss_biases + loss_vars, color='g')
    plt.show()



if __name__ == '__main__':
    #show_gradient_bias()
    show_variance_function()

# -0.036934608722087675
# -0.035532752567470856
# 0.07997541179226468