import tensorflow as tf
import numpy as np

xe_sm_grad = None
rbf_grad = None
y_hot = None

#BackProp Params
num_duds = 0
do_useless_dimensions = False
z_normalized = True
z_bar_normalized = True
tau_normalized = True


@tf.RegisterGradient("stub_and_save")
def _stub_and_save(unused_op, grad):
    global xe_sm_grad
    xe_sm_grad = grad
    return tf.ones(grad.shape)


@tf.RegisterGradient("stub_rbf_grad")
def _stub_rbf_grad(unused_op, grad):
    global rbf_grad
    rbf_grad = grad
    return tf.ones(grad.shape)


def _normalise(grad):
    m, d, K = grad.shape
    K = K.value
    grad_mag = tf.reduce_sum(grad ** 2.0, axis=1) ** 0.5
    normalised = grad / (tf.reshape(grad_mag, [-1, 1, K]) + 10 ** (-70))
    return normalised


def _z_bar_or_tau_grad(grad, do_normalise):
    m, d, K = grad.shape
    K = K.value
    if do_normalise:
        grad = _normalise(grad)
    else:
        # Need to include the rbf gradient if not normalising otherwise
        # furthest points will be parabolically most influential
        grad *= tf.reshape(rbf_grad, [-1, 1, K])
    xe_sm_grad_reshaped = tf.reshape(xe_sm_grad, [-1, 1, K])
    new_grad = xe_sm_grad_reshaped * grad
    return new_grad


@tf.RegisterGradient("z_grad")
def _z_grad(unused_op, grad):
    if z_normalized:
        grad = _normalise(grad)
    m, d, K = grad.shape
    m = m.value
    d = d.value
    K = K.value
    # Experiment with dud points
    ones = tf.ones(shape=(m - num_duds * K, K), dtype=tf.float32)
    zeros = tf.zeros(shape=(num_duds * K, K), dtype=tf.float32)
    duds_mask = tf.concat([ones, zeros], axis=0)
    ones = np.ones(shape=(m, d), dtype=np.float32)
    global y_hot
    y_hot_mask = y_hot * duds_mask * xe_sm_grad
    y_hot_mask = tf.reshape(y_hot_mask, [-1, 1, K])
    all_dim_inds = np.arange(d)
    if do_useless_dimensions:
        # Experiment with useless dimensions
        zero_inds = [np.random.choice(all_dim_inds, d // 2, replace=False) for j in xrange(K)]
        for i in xrange(0, m, K):
            for c in xrange(K):
                zero_inds_c = zero_inds[c]
                ones[i + c][zero_inds_c] = 0.0
            # ones[i + 1][1] = 0.0
        useless_dim_mask = tf.constant(ones)
        reshaped_useless_dim_mask = tf.reshape(useless_dim_mask, [m, d, 1])
        y_hot_mask = reshaped_useless_dim_mask * y_hot_mask
    new_grad = float(d) ** 0.5 * y_hot_mask * grad
    return new_grad


@tf.RegisterGradient("z_bar_grad")
def _z_bar_grad(unused_op, grad):
    m, _, K = grad.shape
    m = m.value
    K = K.value
    return _z_bar_or_tau_grad(grad, z_bar_normalized) / float(m) * 2.0


@tf.RegisterGradient("tau_grad")
def _tau_grad(unused_op, grad):
    m, _, K = grad.shape
    m = m.value
    K = K.value
    normed = _z_bar_or_tau_grad(grad, tau_normalized) / float(m) * 2.0
    return normed


class RBF:

    def __init__(self, conf):
        self.num_class = conf.num_class
        self.n = conf.n
        self.d = conf.d
        train_centres_taus = conf.train_centres_taus
        self.z = tf.get_variable("z", shape=[self.n, self.d],
                                 #initializer=tf.constant_initializer(np.array([[-5.0, -5.0], [-1.0, -1.0], [5.0, 5.0], [-5.5, -5.5], [-1.5, -1.5], [5.5, 5.5]])))
                                 initializer=tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd))
        self.y = tf.placeholder(tf.int32, shape=[self.n], name="y")
        self.rbf_c = conf.rbf_c
        self.z_bar = tf.get_variable("z_bar", shape=[self.d, self.num_class],
                                     initializer=tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd))
        self.tau = tf.abs(tf.get_variable("tau", shape=[self.d, self.num_class],
                                          initializer=tf.constant_initializer(0.5 / float(self.d) ** 0.5 * np.ones(shape=[self.d, self.num_class]))))#initializer=tf.truncated_normal_initializer(stddev=0.5)))

        self.gen_zs = tf.placeholder(dtype=tf.float32, shape=[None, self.d], name='gen_zs')
        self.gen_y = tf.placeholder(dtype=tf.int32, shape=[None], name='gen_y')
        self.ind = tf.placeholder(dtype=tf.float32, shape=[], name='ind')

        _, gen_loss = self._rbf_softmax(self.gen_zs, self.gen_y)
        self.rbf, main_loss = self._rbf_softmax(self.z, self.y)


        if not train_centres_taus:
            var_list = [self.z, self.z_bar]
            self.train_op = conf.optimizer(learning_rate=conf.lr).minimize(main_loss, var_list=var_list)
        else:
            self.train_op = conf.optimizer(learning_rate=conf.lr).minimize(main_loss)
            self.train_gen_op = conf.optimizer(learning_rate=conf.lr).minimize(self.ind * gen_loss)

    def _rbf_softmax(self, z, labels, is_gen=False):
        g = tf.get_default_graph()
        z_re = tf.reshape(z, [-1, self.d, 1])
        z_tile = tf.tile(z_re, [1, 1, self.num_class])
        with g.gradient_override_map({'Identity': "z_grad"}):
            z_identity = tf.identity(z_tile, name='Identity')

        z_bar_re = tf.reshape(self.z_bar, [1, self.d, -1])
        z_bar_tile = tf.tile(z_bar_re, [self.n, 1, 1])
        with g.gradient_override_map({'Identity': "z_bar_grad"}):
            z_bar_identity = tf.identity(z_bar_tile, name='Identity')

        tau_re = tf.reshape(self.tau, [1, self.d, -1])
        tau_tile = tf.tile(tau_re, [self.n, 1, 1])
        with g.gradient_override_map({'Identity': "tau_grad"}):
            tau_identity = tf.identity(tau_tile, name='Identity')

        x_diff = tf.subtract(z_identity, z_bar_identity, name='Sub')
        x_diff_sq = x_diff ** 2.0
        tau_square = tau_identity ** 2.0
        weighted_x_diff_sq = tf.multiply(tau_square, x_diff_sq)
        neg_dist = -tf.reduce_sum(weighted_x_diff_sq, axis=1)

        with g.gradient_override_map({'Identity': "stub_rbf_grad"}):
            neg_dist_identity = tf.identity(neg_dist, name='Identity')
        exp = tf.exp(neg_dist_identity)
        rbf = self.rbf_c * exp

        with g.gradient_override_map({'Identity': "stub_and_save"}):
            rbf_identity = tf.identity(rbf, name='Identity')

        global y_hot
        y_hot = tf.one_hot(labels, self.num_class)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=rbf_identity)
        return rbf, loss


    def all_ops(self):
        return [self.train_op, self.train_gen_op, self.z, self.z_bar, self.tau, tf.nn.softmax(self.rbf)] #xe_sm_grad  self.tau_square, self.x_diff_sq, self.weighted_x_diff_sq, self.neg_dist, self.exp

    def z_bar_tau_ops(self):
        return self.z_bar, self.tau