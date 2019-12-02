import tensorflow as tf
import pdf_functions as pf


class MultivariateKernelService(object):
    """Service that can be used to construct a tf component for training and/or evaluating a multivariate kernel density
    estimator.
    """
    
    def __init__(self, pdf_service, conf, actuals=None):
        self.pdf_service = pdf_service
        self.r = conf.r
        self.d = conf.d
        self.k = conf.k
        self.actuals = actuals
        self.c = conf.c
        self.min_eigenvalue = conf.min_eigenvalue
        self.max_eigenvalue = conf.max_eigenvalue

    def create_tensors(self, lr, batch_size, a, a_star1, a_star2, Q_init, lam_inv_init):
        """Build and return the tensors related to training, reporting and inference for a multivariate kernel density
        estimator.
        """
        # Placeholders
        low_bias_Q = tf.placeholder(dtype=tf.float32, shape=[self.d, self.d], name='low_bias_Q')
        low_bias_lam_inv = tf.placeholder(dtype=tf.float32, shape=[self.d], name='low_bias_lam_inv')
        # Variables
        Q = tf.Variable(Q_init, name='Q', dtype=tf.float32)
        lam_inv = tf.Variable(lam_inv_init, name='lam_inv', dtype=tf.float32)
        threshold = 2 * tf.reduce_sum(lam_inv ** 2)
        # f(a) - our pdf
        fa, eigen_distances = self.pdf_service.eigen_probabilities(lam_inv, a, a_star1, batch_size, Q) #, threshold)
        # If actuals were passed in, train to fit on the actual distribution
        if self.actuals is not None:
            Q_act, lam_inv_act, means = self.actuals
            pa_estimate, _ = self.pdf_service.eigen_probabilities(lam_inv_act, a, means, batch_size, Q_act)
        # Otherwise we have a real problem where the distribution is unknown
        else:
            lb_threshold = 2 * tf.reduce_sum(low_bias_lam_inv ** 2)
            pa_estimate, _ = self.pdf_service.eigen_probabilities(low_bias_lam_inv, a, a_star2, batch_size, low_bias_Q) #, lb_threshold)
        Qt = tf.transpose(Q)
        QtQ = tf.matmul(Qt, Q)
        A = tf.matmul(Q / lam_inv, Qt)
        loss_Q = tf.reduce_mean(eigen_distances)
        loss_lam_inv = tf.reduce_mean((pa_estimate - fa) ** 2)
        reg = tf.reduce_mean((QtQ - tf.eye(self.d)) ** 2)

        # Update Q
        dloss_dQ = tf.gradients(loss_Q, Q)[0]
        d_loss_dlam_inv = tf.gradients(loss_lam_inv, lam_inv)[0]
        dreg_dQ = tf.gradients(reg, Q)[0]
        dloss_dQ_normed = dloss_dQ / tf.norm(dloss_dQ)
        dreg_dQ_normed = dreg_dQ / tf.norm(dreg_dQ)
        Q_step = self.k * dloss_dQ_normed + (1 - self.k) * dreg_dQ_normed
        Q_step = Q_step / tf.norm(Q_step)
        Q_train = tf.assign_sub(Q, lr * Q_step)
        # Update lambda inverse
        lam_inv_step = d_loss_dlam_inv / tf.norm(d_loss_dlam_inv)
        lam_inv_train = tf.assign_sub(lam_inv, lr * lam_inv_step * lam_inv)

        tensors = Q_train, lam_inv_train, Q, lam_inv, QtQ, pa_estimate, fa, A, loss_lam_inv
        return MvKdeGraph(low_bias_Q, low_bias_lam_inv, tensors)


    def create_tensors_lam_only(self, lr, batch_size, a, a_star1, a_star2, low_bias_lam_inv):
        shape = low_bias_lam_inv.shape
        initializer = tf.random_uniform_initializer(minval=self.min_eigenvalue, maxval=self.max_eigenvalue)
        lam_inv = tf.get_variable('lamda_inverse', shape=shape, initializer=initializer)
        threshold = 2 * tf.reduce_sum(lam_inv ** 2)
        fa, eigen_distances = self.pdf_service.eigen_probabilities(lam_inv, a, a_star1, batch_size)  # , threshold)
        lb_threshold = 2 * tf.reduce_sum(low_bias_lam_inv ** 2)
        pa_estimate, _ = pf.eigen_probabilities(low_bias_lam_inv, a, a_star2, batch_size)  # , lb_threshold)
        train, _ = self._lamda_inv_train(pa_estimate, fa, lam_inv, lr)
        return train, lam_inv

    def _lamda_inv_train(self, pa_estimate, fa, lam_inv, lr):
        loss_lam_inv = tf.reduce_mean((pa_estimate - fa) ** 2)
        d_loss_dlam_inv = tf.gradients(loss_lam_inv, lam_inv)[0]
        lam_inv_step = d_loss_dlam_inv / tf.norm(d_loss_dlam_inv)
        lam_inv_train = tf.assign_sub(lam_inv, lr * lam_inv_step * lam_inv)
        return lam_inv_train, loss_lam_inv



class MvKdeGraph(object):
    """Represents a complete multivariate kernel density estimator graph."""

    def __init__(self, low_bias_Q, low_bias_lam_inv, tensors):
        # Placeholders
        self.low_bias_Q = low_bias_Q
        self.low_bias_lam_inv = low_bias_lam_inv
        # Tensors used for training and reporting
        self.tensors = tensors
