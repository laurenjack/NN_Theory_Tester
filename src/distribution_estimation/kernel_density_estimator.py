import tensorflow as tf


class KernelDensityEstimator(object):
    """Represents a Kernel Density Estimate with variable bandwidth h, over a set of reference samples a_star.
    """

    def __init__(self, conf):
        self.h = tf.Variable(2.0, name='h')
        self.m = conf.m
        self.r = conf.r
        self.lr = conf.lr
        self.a = tf.placeholder(dtype=tf.float32, shape=[self.m], name='a')
        self.a_star = tf.placeholder(dtype=tf.float32, shape=[self.r], name='a_star')

    def expected_likelihood(self):
        """Given a set of m observations: a, and a set of r reference observation a_star drawn from the same empirical
        distribution (which are used to give the Kernel Density Estimate), compute the expected likelihood over a.

        This function returns a tensor to be maximised with respect to h (the bandwidth of the kernel function).
        """
        m = self.a.shape[0].value
        r = self.a_star.shape[0].value
        a_difference = tf.reshape(self.a, [m, 1]) - tf.reshape(self.a_star, [1, r])
        a_distance_squared = a_difference ** 2.0
        exponent = a_distance_squared / (2.0 * self.h ** 2.0)
        kernel = 1.0 / self.h * tf.exp(-exponent)
        expected_likelihood = tf.reduce_mean(kernel)
        train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(-expected_likelihood)
        return train_op, self.h