import tensorflow as tf
import numpy as np
import configuration

conf = configuration.get_conf()
xe_sm_grad = None
variance_grad = None
rbf_grad = None
y_hot = None

z_grad = None
z_bar_grad = None
tau_grad = None

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
    normalised = grad / (tf.reshape(grad_mag, [-1, 1, K]) + conf.norm_epsilon)
    return normalised


@tf.RegisterGradient("z_grad")
def _z_grad(unused_op, grad):
    grad = _normalise(grad)
    m, d, K = grad.shape
    m = m.value
    d = d.value
    K = K.value
    # Experiment with dud points
    ones = tf.ones(shape=(m - conf.num_duds * K, K), dtype=tf.float32)
    zeros = tf.zeros(shape=(conf.num_duds * K, K), dtype=tf.float32)
    duds_mask = tf.concat([ones, zeros], axis=0)
    ones = np.ones(shape=(m, d), dtype=np.float32)
    global y_hot
    y_hot_mask = y_hot * duds_mask * xe_sm_grad
    y_hot_mask = tf.reshape(y_hot_mask, [-1, 1, K])
    all_dim_inds = np.arange(d)
    if conf.do_useless_dimensions:
        # Experiment with useless dimensions
        zero_inds = [np.random.choice(all_dim_inds, d // 2, replace=False) for j in xrange(K)]
        for i in xrange(0, m, K):
            for c in xrange(K):
                zero_inds_c = zero_inds[c]
                ones[i + c][zero_inds_c] = 0.0
        useless_dim_mask = tf.constant(ones)
        reshaped_useless_dim_mask = tf.reshape(useless_dim_mask, [m, d, 1])
        y_hot_mask = reshaped_useless_dim_mask * y_hot_mask
    new_grad = float(d) ** 0.5 * y_hot_mask * grad
    global z_grad
    z_grad = new_grad
    return new_grad


@tf.RegisterGradient("z_bar_grad")
def _z_bar_grad(unused_op, grad):
    m, d, K = grad.shape
    m = m.value
    K = K.value
    grad = _normalise(grad)
    xe_sm_grad_reshaped = tf.reshape(xe_sm_grad, [-1, 1, K])
    grad = xe_sm_grad_reshaped * grad
    global z_bar_grad
    z_bar_grad = grad / float(m) ** 0.5
    return z_bar_grad


@tf.RegisterGradient("tau_grad")
def _tau_grad(unused_op, grad):
    m, d, K = grad.shape
    m = m.value
    d = d.value
    K = K.value
    global tau_grad
    grad = tf.reshape(variance_grad, shape=[1, d, K]) * grad
    tau_grad = tf.sign(grad) * tf.abs(grad / float(m)) ** 0.5  * 0.5
    return tau_grad


class RBF:

    def __init__(self, z, z_bar_init, tau_init):
        self.num_class = conf.num_class
        self.n = conf.n
        self.d = conf.d
        self.z = z
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
            self.normalise_tau = (conf.target_variance - weighted_variance_id) ** 2.0
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
        self.train_op = conf.optimizer(learning_rate=conf.lr).minimize(loss)

    def all_ops(self):
        return [self.train_op, self.z, self.z_bar, self.tau, tf.nn.softmax(self.rbf)]

    def test_ops(self):
        return self.train_op, self.x_diff_sq, self.tau_square, self.weighted_x_diff_sq,\
               self.weighted_x_diff_sq_other, self.normalise_tau, self.tau, tau_grad, variance_grad,\
               self.sm, z_grad, z_bar_grad

    def variable_ops(self):
        return self.z, self.z_bar, self.tau