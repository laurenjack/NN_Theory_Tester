import tensorflow as tf
import pdf_functions as pf


class MultivariateKernelService(object):
    """Service that can be used to construct a tf component for training and/or evaluating a multivariate kernel density
    estimator.
    """
    
    def __init__(self, conf, actuals=None):
        self.r = conf.r
        self.d = conf.d
        self.k = conf.k
        self.Q_init = conf.Q_init
        self.lam_inv_init = conf.lam_inv_init
        self.actuals = actuals

    def create_tensors(self, lr, batch_size, a, a_star1, a_star2, low_bias_Q, low_bias_lam_inv):
        """Build and return the tensors related to training, reporting and inference for a multivariate kernel density
        estimator.
        """
        # Variables
        Q = tf.Variable(self.Q_init, name='Q', dtype=tf.float32)
        lam_inv = tf.Variable(self.lam_inv_init, name='lam_inv', dtype=tf.float32)
        # f(a) - our pdf
        log_fa, exponentials, eigen_distances, difference = pf.sum_of_log_eigen_probs(Q, lam_inv, a, a_star1, batch_size)
        # If actuals were passed in, train to fit on the actual distribution
        if self.actuals is not None:
            Q_act, lam_inv_act, means = self.actuals
            pa_estimate, log_pa = pf.normal_exponent(a, means, Q_act, lam_inv_act, batch_size)
        # Otherwise we have a real problem where the distribution is unknown
        else:
            pa_estimate, log_pa = pf.sum_of_log_eigen_probs(low_bias_Q, low_bias_lam_inv, a, a_star2, batch_size)
        Qt = tf.transpose(Q)
        QtQ = tf.matmul(Qt, Q)
        A = tf.matmul(Q / lam_inv, Qt)
        # inv_log_fa = 1.0 / log_fa
        # inv_log_pa = 1.0 / log_pa
        loss = tf.reduce_mean((log_pa - log_fa) ** 2) # tf.reduce_mean((pa_estimate - fa) ** 2)
        # chi_square = pf.chi_squared_distribution(self.d, tf.reduce_sum(eigen_distances, axis=2))
        # chi_square_sum = tf.reshape(tf.reduce_sum(chi_square, axis=1), [batch_size, 1])
        normal = tf.exp(self.d -0.5 * tf.reduce_sum(eigen_distances, axis=2))
        normal_sum = tf.reshape(tf.reduce_sum(normal, axis=1), [batch_size, 1])
        weights = normal / normal_sum
        # weights = tf.reshape(weights, [batch_size, self.r, 1])
        weights = exponentials / tf.reshape(tf.reduce_sum(exponentials, axis=1), [batch_size, 1, self.d])
        log_delta = log_fa - log_pa
        high_log = tf.nn.top_k(log_delta, k=10)
        reg = tf.reduce_mean((QtQ - tf.eye(self.d)) ** 2)

        # Update Q
        # dloss_dQ, d_loss_dlam_inv = tf.gradients(loss, [Q, lam_inv])
        dloss_dQ, d_loss_dlam_inv = pf.gradients_with_flex_weights(log_delta, difference, Q, lam_inv, weights,
                                                                   batch_size)
        dreg_dQ = tf.gradients(reg, Q)[0]
        dloss_dQ_normed = dloss_dQ / tf.norm(dloss_dQ)
        dreg_dQ_normed = dreg_dQ / tf.norm(dreg_dQ)
        Q_step = self.k * dloss_dQ_normed + (1 - self.k) * dreg_dQ_normed
        Q_step = Q_step / tf.norm(Q_step)
        Q_train = tf.assign_sub(Q, lr * Q_step)
        # Update lambda inverse
        lam_inv_step = d_loss_dlam_inv / tf.norm(d_loss_dlam_inv)
        lam_inv_train = tf.assign_sub(lam_inv, lr * lam_inv_step * lam_inv)

        return Q_train, lam_inv_train, Q, lam_inv, QtQ, pa_estimate, log_fa, A, high_log # A, loss


class MvKdeGraph(object):
    """Represents a complete multivariate kernel density estimator graph."""

    def __init__(self, conf, multivariate_kernel_service):
        # Placeholders
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.a = tf.placeholder(dtype=tf.float32, shape=[None, conf.d], name='a')
        self.a_star1 = tf.placeholder(dtype=tf.float32, shape=[conf.r, conf.d], name='a_star1')
        self.a_star2 = tf.placeholder(dtype=tf.float32, shape=[conf.r, conf.d], name='a_star2')
        self.low_bias_Q = tf.placeholder(dtype=tf.float32, shape=[conf.d, conf.d], name='low_bias_Q')
        self.low_bias_lam_inv = tf.placeholder(dtype=tf.float32, shape=[conf.d], name='low_bias_lam_inv')
        # Tensors used for training and reporting
        self.tensors = multivariate_kernel_service.create_tensors(self.lr, self.batch_size, self.a, self.a_star1,
                                                                  self.a_star2, self.low_bias_Q, self.low_bias_lam_inv)
