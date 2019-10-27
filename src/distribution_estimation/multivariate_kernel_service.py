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
        fa, true_exp = pf.eigen_probabilities(Q, lam_inv, a, a_star1, batch_size)
        # If actuals were passed in, train to fit on the actual distribution
        if self.actuals is not None:
            Q_act, lam_inv_act, means = self.actuals
            pa_estimate, _ = pf.eigen_probabilities(Q_act, lam_inv_act, a, means, batch_size)
        # Otherwise we have a real problem where the distribution is unknown
        else:
            pa_estimate, _ = pf.eigen_probabilities(low_bias_Q, low_bias_lam_inv, a, a_star2, batch_size)
        Qt = tf.transpose(Q)
        QtQ = tf.matmul(Qt, Q)
        A = tf.matmul(Q / lam_inv , Qt)
        # Q_norm = tf.reduce_sum(Q ** 2, axis=1) ** 0.5
        # exponent = tf.reduce_mean(eigen_distance, axis=1)
        loss_Q = -tf.reduce_mean(tf.reduce_prod(fa *2.73, axis=1)) #-tf.reduce_mean(tf.exp(-true_exp)) #-tf.reduce_mean(tf.exp(-exponent / 2.0))
        loss_lam = tf.reduce_mean((pa_estimate - fa) ** 2)
        reg = tf.reduce_mean((QtQ - tf.eye(self.d)) ** 2)

        # Update Q
        dloss_dQ = tf.gradients(loss_Q, Q)[0]
        dreg_dQ = tf.gradients(reg, Q)[0]
        dloss_dQ_normed = dloss_dQ / tf.norm(dloss_dQ)
        dreg_dQ_normed = dreg_dQ / tf.norm(dreg_dQ)
        Q_step = self.k * dloss_dQ_normed + (1 - self.k) * dreg_dQ_normed
        Q_step = Q_step / tf.norm(Q_step)
        Q_train = tf.assign_sub(Q, lr * Q_step)
        # Update lambda inverse
        d_loss_dlam_inv = tf.gradients(loss_lam, lam_inv)[0]
        lam_inv_step = d_loss_dlam_inv / tf.norm(d_loss_dlam_inv)
        lam_inv_train = tf.assign_sub(lam_inv, lr * lam_inv_step * lam_inv)

        return Q_train, lam_inv_train, Q, lam_inv, QtQ, pa_estimate, fa, A


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
