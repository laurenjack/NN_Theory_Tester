import tensorflow as tf
import numpy as np

xe_sm_grad = None
variance_grad = None
pre_tau_grad = None
rbf_grad = None
y_hot = None

global final_grad
normed_grad = None
z_grad = None
z_bar_grad = None

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

@tf.RegisterGradient("stub_and_save_variance_grad")
def _stub_and_save_variance_grad(unused_op, grad):
    global variance_grad
    variance_grad = grad
    return tf.ones(grad.shape)


@tf.RegisterGradient("stub_rbf_grad")
def _stub_rbf_grad(unused_op, grad):
    global rbf_grad
    rbf_grad = grad
    return tf.ones(grad.shape)

@tf.RegisterGradient("zero_the_grad")
def _zero_the_grad(unused_op, grad):
    return tf.zeros(grad.shape)


def _normalise(grad):
    m, d, K = grad.shape
    K = K.value
    grad_mag = tf.reduce_sum(grad ** 2.0, axis=1) ** 0.5
    normalised = grad / (tf.reshape(grad_mag, [-1, 1, K])+ 10 ** (-70))
    return normalised


def _z_bar_or_tau_grad(grad, do_normalise):
    m, d, K = grad.shape
    m = m.value
    K = K.value
    if do_normalise:
        grad = _normalise(grad)
    else:
        # Need to include the rbf gradient if not normalising otherwise
        # furthest points will be parabolically most influential
        grad *= tf.reshape(rbf_grad, [-1, 1, K])
    xe_sm_grad_reshaped = tf.reshape(xe_sm_grad, [-1, 1, K])
    new_grad = xe_sm_grad_reshaped * grad
    return new_grad / float(m) ** 0.5


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
    global z_grad
    z_grad = new_grad
    return new_grad

# @tf.RegisterGradient("debug")
# def _debug(unused_op, grad):
#     global norm_tau_debug
#     norm_tau_debug = grad
#     return grad


@tf.RegisterGradient("z_bar_grad")
def _z_bar_grad(unused_op, grad):
    new_grad = _z_bar_or_tau_grad(grad, z_bar_normalized)
    global z_bar_grad
    z_bar_grad = new_grad
    return new_grad


@tf.RegisterGradient("tau_grad")
def _tau_grad(unused_op, grad):
    m, d, K = grad.shape
    m = m.value
    d = d.value
    K = K.value
    global pre_tau_grad
    pre_tau_grad = grad
    # reshaped_y_hot = tf.reshape((1.0 - y_hot), [-1, 1, K])
    # grad_mag = tf.reduce_sum(grad ** 2.0, axis=1) ** 0.5
    # normalised = grad / (tf.reshape(grad_mag, [-1, 1, K]) + reshaped_y_hot + 10 ** -70)
    # global normed_grad
    # normed_grad = normalised
    global final_grad
    final_grad = tf.reshape(variance_grad, shape=[1, d, K]) * grad
    final_grad = tf.sign(final_grad) * tf.abs(final_grad) ** 0.5 / float(m) / float(d)
    return final_grad


class RBF:

    # tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd)
    # tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd)
    #tf.constant_initializer(0.5 / float(self.d) ** 0.5 * np.ones(shape=[self.d, self.num_class])))

    def __init__(self, conf, z_init, z_bar_init, tau_init):
        self.num_class = conf.num_class
        self.n = conf.n
        self.d = conf.d
        train_centres_taus = conf.train_centres_taus
        self.z = tf.get_variable("z", shape=[self.n, self.d],
                                 #initializer=tf.constant_initializer(np.array([[-5.0, -5.0], [-1.0, -1.0], [5.0, 5.0], [-5.5, -5.5], [-1.5, -1.5], [5.5, 5.5]])))
                                 initializer=z_init)
        self.y = tf.placeholder(tf.int32, shape=[self.n], name="y")
        self.rbf_c = conf.rbf_c
        self.z_bar = tf.get_variable("z_bar", shape=[self.d, self.num_class],
                                     initializer=z_bar_init)
        self.tau = tf.abs(tf.get_variable("tau", shape=[self.d, self.num_class],
                                          initializer=tau_init))#initializer=tf.truncated_normal_initializer(stddev=0.5)))

        self.tau_square = self.tau ** 2.0

        global y_hot
        y_hot = tf.one_hot(self.y, self.num_class)

        g = tf.get_default_graph()
        z_re = tf.reshape(self.z, [-1, self.d, 1])
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
        self.x_diff_sq = x_diff ** 2.0
        self.tau_square_tile = tau_identity ** 2.0

        with g.gradient_override_map({'Identity': "zero_the_grad"}):
            x_diff_sq_id = tf.identity(self.x_diff_sq, name='Identity')
            self.weighted_x_diff_sq_other = tf.multiply(self.tau_square_tile, x_diff_sq_id, name="wxds_other")
            filtered_sum = tf.reshape(y_hot, [self.n, 1, self.num_class]) * self.weighted_x_diff_sq_other
            self.weighted_variance = tf.reduce_sum(filtered_sum, axis=0) / tf.reshape(tf.reduce_sum(y_hot, axis=0), [1, self.num_class])
            # TODO there will be an issue here when a class is not present in the batch
        with g.gradient_override_map({'Identity': "stub_and_save_variance_grad"}):
            weighted_variance_id = tf.identity(self.weighted_variance, name='Identity')
            self.normalise_tau = (1.0 - weighted_variance_id) ** 2.0
            tau_loss = tf.reduce_sum(self.normalise_tau)

        with g.gradient_override_map({'Identity': "zero_the_grad"}):
            tau_sq_id = tf.identity(self.tau_square_tile, name='Identity')
            self.weighted_x_diff_sq = tf.multiply(tau_sq_id, self.x_diff_sq)
            self.neg_dist = -tf.reduce_mean(self.weighted_x_diff_sq, axis=1)

        with g.gradient_override_map({'Identity': "stub_rbf_grad"}):
            neg_dist_identity = tf.identity(self.neg_dist, name='Identity')
        self.exp = tf.exp(neg_dist_identity)
        self.rbf = self.rbf_c * self.exp
        with g.gradient_override_map({'Identity': "stub_and_save"}):
            rbf_identity = tf.identity(self.rbf, name='Identity')
        self.sm = tf.nn.softmax(rbf_identity)

        # loss = -tf.reduce_mean(tf.reduce_sum(y_hot*tf.log(sm), axis=1))
        self.main_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=rbf_identity)
        loss = tf.reduce_sum(self.main_loss) + tau_loss
        if not train_centres_taus:
            var_list = [self.z, self.z_bar]
            self.train_op = conf.optimizer(learning_rate=conf.lr).minimize(loss, var_list=var_list)
        else:
            self.train_op = conf.optimizer(learning_rate=conf.lr).minimize(loss)

    def all_ops(self):
        return [self.train_op, self.z, self.z_bar, self.tau, tf.nn.softmax(self.rbf)] #xe_sm_grad  self.tau_square, self.x_diff_sq, self.weighted_x_diff_sq, self.neg_dist, self.exp

    def test_ops(self):
        return self.train_op, self.x_diff_sq, self.tau_square, self.weighted_x_diff_sq,\
               self.weighted_x_diff_sq_other, self.normalise_tau, self.tau, normed_grad, final_grad, variance_grad,\
               self.sm, z_grad, z_bar_grad

    def variable_ops(self):
        self.z, self.z_bar, self.tau