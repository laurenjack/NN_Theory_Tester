import math

import tensorflow as tf
import numpy as np


class KernelDensityEstimator(object):
    """Represents a Kernel Density Estimator with variable bandwidth A, over a set of reference samples a_star.

    This class can be used to cosntruct the entire training graph for the bandwidth, or a sub-graph for computing
    f(a) for a given bandwidth.
    """

    def __init__(self, conf, data_generator=None):
        self.data_generator = data_generator
        self.r = conf.r
        self.d = conf.d
        self.R_init = conf.R_init
        self.lr = tf.placeholder(dtype=conf.float_precision, shape=[], name='lr')
        self.float_precision = conf.float_precision
        self.fit_to_underlying_pdf = conf.fit_to_underlying_pdf
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.a = tf.placeholder(dtype=conf.float_precision, shape=[None, self.d], name='a')
        self.a_star1 = tf.placeholder(dtype=conf.float_precision, shape=[self.r, self.d], name='a_star1')
        self.a_star2 = tf.placeholder(dtype=conf.float_precision, shape=[self.r, self.d], name='a_star2')
        # A [d, d] array. This is the placeholder for the exogenous low bias bandwidth which is injected into the graph
        # at each training step. It is injected as an inverse, because the matrix A in the kernel function is never
        # required in a form other than it's inverse.
        self.low_bias_A_inverse = tf.placeholder(dtype=conf.float_precision, shape=[self.d, self.d],
                                                 name='low_bias_A_inverse')

    def construct_kde_training_graph(self):
        """This function creates a tensorflow graph to optimise A = RtR, the bandwidth of the kernel density estimator.

        Returns:
            An instance of KdeTensors - see constructor doc
        """
        R_inverse = tf.Variable(np.linalg.inv(self.R_init), name='R_inverse', dtype=self.float_precision)
        A_inverse = tf.matmul(R_inverse, tf.transpose(R_inverse))
        fa = self.pdf(A_inverse, self.a_star1)
        # If a data generator was passed in, use the actual distribution from the data generator:
        if self.fit_to_underlying_pdf:
           pa_estimate = self.data_generator.pdf(self.a, self.batch_size)
        # Otherwise we have a real problem where the distribution is unknown
        else:
            pa_estimate = self.pdf(self.low_bias_A_inverse, self.a_star2)
        pa_estimate = pa_estimate
        loss = 0.5 * tf.reduce_mean((fa - pa_estimate) ** 2.0)
        optimiser = tf.train.GradientDescentOptimizer(self.lr)
        gradient_var_pairs = optimiser.compute_gradients(loss)
        new_gradient_var_pairs = []
        for gradient, var in gradient_var_pairs:
            new_gradient = gradient / tf.reduce_mean(tf.abs(gradient))
            new_gradient_var_pairs.append((new_gradient, var))
        train = optimiser.apply_gradients(new_gradient_var_pairs)
        return KdeTensors(train, loss, pa_estimate, fa, A_inverse, tf.matrix_inverse(A_inverse))

    def pdf(self, A_inverse, a_star):
        """Compute f(a) for the [batch_size, d] set of points a, using the [r, d] set of reference points and the
        inverse bandwitdth matrix A_inverse.

        Args:
            A_inverse: A [d, d] tensor, the bandwidth of the kernel density estimate.

        Returns:
            A [batch_size] tensor. The relative likelihood f(a) for each element of a.
        """
        H_inverse = tf.matmul(A_inverse, tf.transpose(A_inverse))
        difference = tf.reshape(self.a, [self.batch_size, 1, self.d]) - tf.reshape(a_star, [1, self.r, self.d])
        distance_squared = tf.reshape(tf.tensordot(difference, H_inverse, axes=[[2], [0]]),
                                      [self.batch_size, self.r, 1, self.d])
        distance_squared = tf.matmul(distance_squared, tf.reshape(difference, [self.batch_size, self.r, self.d, 1]))
        # Drop one of the unnecessary 1 dimensions, leave the other for future broadcasting.
        distance_squared = tf.reshape(distance_squared, [self.batch_size, self.r, 1])
        exponent = 0.5 * (-distance_squared)
        kernel = tf.exp(exponent)
        det_A_inverse = tf.matrix_determinant(A_inverse)
        kernel = det_A_inverse * kernel  / (2.0 * math.pi) ** (self.d * 0.5)
        kernel = kernel #** (1.0 / float(self.d))
        fa = tf.reduce_mean(tf.reshape(kernel, [self.batch_size, self.r]), axis=1)
        # fa = det_A_inverse * fa_unscaled  / (2.0 * math.pi) ** (self.d * 0.5)
        return fa


class KdeTensors:
    """Represents all the tensors which we would like to either optimise, or extract the value for, from the Kde graph:
    """

    def __init__(self, train, loss, pa_estimate, fa, A_inverse, A):
        """
        Args:
            train: A gradient descent optimiser but scaled by the mean absolute value of the [d, d] gradient.
            loss: The loss function J = sum(f*(x) - f(x))^2 or in the known case: Jp = sum(p(x) - f(x))^2
            pa_estimate: The [m] tensor f*(x) or in the known case p(x), i.e. m relative likelihoods from a pdf
            fa: The [m] tensor f(x), i.e. the m relative likelihoods produced by the kernel density estimator
            A_inverse: The inverse of the bandwidth A, the representation required to compute f(a) in the graph
            A: The bandwidth A
        """
        self.train = train
        self.loss = loss
        self.pa_estimate = pa_estimate
        self.fa = fa
        self.A_inverse = A_inverse
        self.A = A
