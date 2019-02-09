import tensorflow as tf


class KernelDensityEstimator(object):
    """Represents a Kernel Density Estimate with variable bandwidth h, over a set of reference samples a_star.
    """

    def __init__(self, conf):
        self.h = tf.Variable(conf.h_init, name='h', dtype=conf.float_precision)
        self.m = conf.m
        self.r = conf.r
        self.lr = conf.lr
        self.a = tf.placeholder(dtype=conf.float_precision, shape=[self.m], name='a')
        self.a_star = tf.placeholder(dtype=conf.float_precision, shape=[self.r], name='a_star')

    def squared_weighted_mean_error(self):
        """This function returns a tensor to be minimised with respect to h (the bandwidth of the kernel function).

        Given a set of m observations: a, and a set of r reference observation a_star drawn from the same empirical
        distribution (which are used to give the Kernel Density Estimate), compute the squared weighted mean error,
        for each reference observation in a_star.
        """
        m = self.a.shape[0].value
        r = self.a_star.shape[0].value
        a_difference = tf.reshape(self.a, [m, 1]) - tf.reshape(self.a_star, [1, r])
        a_distance_squared = a_difference ** 2.0
        exponent = a_distance_squared / (2.0 * self.h ** 2.0)
        # No h or r as these cancel in the cost function
        kernel = tf.exp(-exponent)
        fa = tf.reduce_mean(kernel, axis=1)
        weighted_error = kernel * a_difference / self.h / tf.reshape(fa, [m, 1])
        weighted_mean_error = tf.reduce_mean(weighted_error, axis=0)
        squared_weighted_mean_error = 0.5 * tf.reduce_sum(weighted_mean_error ** 2.0)
        train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(squared_weighted_mean_error)
        return train_op, squared_weighted_mean_error, self.h, tf.gradients(squared_weighted_mean_error, self.h)