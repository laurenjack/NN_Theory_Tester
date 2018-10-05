import tensorflow as tf

from configuration import conf

xe_sm_grad = None
variance_grad = None
rbf_grad = None
y_hot = None
batch_size = None
z_bar_sd = None

z_grad = None
z_bar_grad = None
tau_grad = None
zds = None

@tf.RegisterGradient("stub_and_save")
def _stub_and_save(unused_op, grad):
    global xe_sm_grad
    xe_sm_grad = grad
    return tf.ones(tf.shape(grad))

@tf.RegisterGradient("stub_and_save_variance_grad")
def _stub_and_save_variance_grad(unused_op, grad):
    global variance_grad
    variance_grad = grad
    return tf.ones(tf.shape(grad))


@tf.RegisterGradient("stub_rbf_grad")
def _stub_rbf_grad(unused_op, grad):
    global rbf_grad
    rbf_grad = grad
    return tf.ones(tf.shape(grad))

@tf.RegisterGradient("zero_the_grad")
def _zero_the_grad(unused_op, grad):
    return tf.zeros(tf.shape(grad))


def _normalise(grad):
    m, d, K = grad.shape
    K = K.value
    grad_mag = tf.reduce_sum(tf.abs(grad), axis=1) # tf.reduce_sum(grad ** 2.0, axis=1) ** 0.5
    normalised = grad / (tf.reshape(grad_mag, [-1, 1, K]) + conf.norm_epsilon)
    return normalised

@tf.RegisterGradient("z_bar_base_grad")
def _z_bar_base_grad(unused_op, grad):
    """
    This gradient is scaled by the standard deviation of z_bar_base, we don't neccessarily want that scaling,
    so all we are doing here is getting rid of it
    """
    global z__bar_base_grad
    z__bar_base_grad = grad * (z_bar_sd + 10.0 ** -8)
    return z__bar_base_grad


@tf.RegisterGradient("z_grad")
def _z_grad(unused_op, grad):
    grad = _normalise(grad)
    _, d, K = grad.shape
    d = d.value
    K = K.value
    global y_hot
    y_hot_mask = xe_sm_grad - (1.0 - y_hot)*tf.maximum(xe_sm_grad, -0.1) # y_hot * xe_sm_grad
    y_hot_mask = tf.reshape(y_hot_mask, [-1, 1, K])
    new_grad =  y_hot_mask * grad # TODO quick test here  float(d) ** 0.5 * # / tf.cast(batch_size, tf.float32) ** 0.5
    global z_grad
    z_grad = new_grad
    return new_grad


@tf.RegisterGradient("z_bar_grad")
def _z_bar_grad(unused_op, grad):
    _, d, K = grad.shape
    K = K.value
    grad = _normalise(grad)
    xe_sm_grad_reshaped = tf.reshape(xe_sm_grad, [-1, 1, K])
    grad = xe_sm_grad_reshaped * grad
    global z_bar_grad
    z_bar_grad = grad / tf.cast(batch_size, tf.float32) ** 0.5
    return z_bar_grad * conf.z_bar_lr_increase_factor


@tf.RegisterGradient("tau_grad")
def _tau_grad(unused_op, grad):
    return tf.sign(variance_grad) * tf.abs(variance_grad) ** 0.5 / (tf.reduce_max(zds, axis=0) + 10.0 ** -5.0) * conf.tau_lr_increase_factor

    # _, d, K = grad.shape
    # # m = m.value
    # d = d.value
    # K = K.value
    # global tau_grad
    # #grad = tf.reshape(variance_grad, shape=[1, d, K]) * grad
    # grad = tf.tile(variance_grad, shape=[batch_size, 1, 1])
    # tau_grad = tf.sign(grad) * tf.abs(grad / tf.cast(batch_size, tf.float32)) ** 0.5
    # return tau_grad * conf.tau_lr_increase_factor


class RbfOps:

    def __init__(self, z, z_bar, tau, a, z_diff, z_diff_sq, tau_square, weighted_z_diff_sq,
               weighted_z_diff_sq_other, target_tau_diff, tau_grad, variance_grad,
               z_grad, z_bar_grad, loss, rbf, weighted_variance, neg_dist):
        self.z = z
        self.z_bar = z_bar
        self.tau = tau
        self.a = a
        self.z_diff = z_diff
        self.z_diff_sq = z_diff_sq
        self.tau_sq = tau_square
        self.wzds = weighted_z_diff_sq
        self.wzdso = weighted_z_diff_sq_other
        self.target_tau_diff = target_tau_diff
        self.tau_grad = tau_grad
        self.variance_grad = variance_grad
        self.z_grad = z_grad
        self.z_bar_grad = z_bar_grad
        self.loss = loss
        self.rbf = rbf
        self.weighted_variance = weighted_variance
        self.neg_dist = neg_dist

    def core_ops(self):
        return [self.loss, self.z, self.z_bar, self.tau, self.a, self.rbf, self.weighted_variance, self.z_diff,
                self.neg_dist, self.wzds]


class RBF:

    def __init__(self, z_bar_init, tau_init, batch_inds=None):
        self.z_bar_init = z_bar_init
        self.tau_init = tau_init
        self.batch_inds = batch_inds
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
        global batch_size
        batch_size = self.batch_size

    def create_all_ops(self, z):
        num_class = conf.num_class
        d = conf.d

        z_batch = z
        if self.batch_inds is not None:
            z_batch = tf.gather(z, self.batch_inds)

        rbf_c = conf.rbf_c
        g = tf.get_default_graph()

        z_bar_base = tf.get_variable("z_bar_base", shape=[d, num_class],
                                     initializer=self.z_bar_init)
        with g.gradient_override_map({'Identity': "z_bar_base_grad"}):
            z_bar_base = tf.identity(z_bar_base, name='Identity')
        z_bar_mew = tf.reduce_mean(z_bar_base, axis=1)
        z_bar_mew = tf.reshape(z_bar_mew, shape=[d, 1])
        z_bar_diff = z_bar_base - z_bar_mew
        global z_bar_sd
        z_bar_sd = tf.reduce_mean(z_bar_diff ** 2.0, axis=1) ** 0.5  # tf.reduce_mean(tf.abs(z_bar_diff), axis=1)
        z_bar_sd = tf.reshape(z_bar_sd, [d, 1])
        z_bar = 5.0 * z_bar_diff / (z_bar_sd + 10.0 ** -8)

        tau = tf.abs(tf.get_variable("tau", shape=[d, num_class],
                                          initializer=self.tau_init))
        with g.gradient_override_map({'Identity': "tau_grad"}):
            tau_identity = tf.identity(tau, name='Identity')
        tau_square = tau_identity ** 2.0

        self.y_hot = tf.one_hot(self.y, num_class)
        global y_hot
        y_hot = self.y_hot

        z_re = tf.reshape(z_batch, [-1, d, 1])
        z_tile = tf.tile(z_re, [1, 1, num_class])
        with g.gradient_override_map({'Identity': "z_grad"}):
            z_identity = tf.identity(z_tile, name='Identity')

        z_bar_re = tf.reshape(z_bar, [1, d, -1])
        tile_shape = tf.concat([[self.batch_size], [1, 1]], axis=0)
        z_bar_tile = tf.tile(z_bar_re, tile_shape)
        with g.gradient_override_map({'Identity': "z_bar_grad"}):
            z_bar_identity = tf.identity(z_bar_tile, name='Identity')

        tau_re = tf.reshape(tau_identity, [1, d, -1])
        tau_tile = tf.tile(tau_re, tile_shape)
        tau_tile_identity = tf.identity(tau_tile, name='Identity')

        z_diff = tf.subtract(z_identity, z_bar_identity, name='Sub')
        z_diff_sq = z_diff ** 2.0
        global zds
        zds = z_diff_sq
        tau_square_tile = tau_tile_identity ** 2.0

        with g.gradient_override_map({'Identity': "zero_the_grad"}):
            z_diff_sq_id = tf.identity(z_diff_sq, name='Identity')
            weighted_z_diff_sq_other = tf.multiply(tau_square_tile, z_diff_sq_id, name="wzds_other") # / float(conf.d)
            fs_shape = tf.concat([[self.batch_size], [1, num_class]], axis=0)
            filtered_sum = tf.reshape(self.y_hot, fs_shape) * weighted_z_diff_sq_other
            class_wise_batch_size = tf.reduce_sum(self.y_hot, axis=0)
            is_greater_than_zero = tf.greater(class_wise_batch_size, 0.01)
            ones = tf.ones([num_class])
            safe_class_wise_batch_size = tf.where(is_greater_than_zero, class_wise_batch_size, ones)
            safe_class_wise_batch_size = tf.reshape(safe_class_wise_batch_size, [1, num_class])
            weighted_variance = tf.reduce_sum(filtered_sum, axis=0) / safe_class_wise_batch_size
        with g.gradient_override_map({'Identity': "stub_and_save_variance_grad"}):
            weighted_variance_id = tf.identity(weighted_variance, name='Identity')
            target_tau_diff = (conf.target_precision - weighted_variance_id) ** 2.0
            tau_loss = tf.reduce_sum(target_tau_diff)

        with g.gradient_override_map({'Identity': "zero_the_grad"}):
            tau_sq_id = tf.identity(tau_square_tile, name='Identity')
            weighted_z_diff_sq = tf.multiply(tau_sq_id, z_diff_sq)
            neg_dist = -tf.reduce_mean(weighted_z_diff_sq, axis=1)

        with g.gradient_override_map({'Identity': "stub_rbf_grad"}):
            neg_dist_identity = tf.identity(neg_dist, name='Identity')
        exp = tf.exp(neg_dist_identity)
        rbf = rbf_c * exp
        with g.gradient_override_map({'Identity': "stub_and_save"}):
            rbf_identity = tf.identity(rbf, name='Identity')
        sm = tf.nn.softmax(rbf_identity)

        # loss = -tf.reduce_mean(tf.reduce_sum(self.y_hot*tf.log(sm), axis=1))
        image_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=rbf_identity)
        loss = tf.reduce_sum(image_loss) + tau_loss

        return RbfOps(z, z_bar, tau, sm, z_diff, z_diff_sq, tau_square, weighted_z_diff_sq,
                weighted_z_diff_sq_other, target_tau_diff, tau_grad, variance_grad,
                z_grad, z_bar_grad, loss, rbf, weighted_variance, neg_dist)

    def create_ops(self, z):
        rbf_ops = self.create_all_ops(z)
        return [rbf_ops.a, rbf_ops.loss, rbf_ops.z, rbf_ops.z_bar, rbf_ops.tau, rbf_ops.rbf]
