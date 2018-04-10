import tensorflow as tf
import numpy as np

xe_sm_grad = None
xe_grad = None

@tf.RegisterGradient("stub_and_save")
def _stub_and_save(unused_op, grad):
    global xe_sm_grad
    xe_sm_grad = grad
    return tf.ones(grad.shape)

@tf.RegisterGradient("normalised")
def _normalised(unused_op, grad):
    m, d, K = grad.shape
    K = K.value
    grad_mag = tf.reduce_sum(grad ** 2.0, axis=1) ** 0.5
    normalised = grad / tf.reshape(grad_mag, [-1, 1, K])
    shaped_xe_sm = tf.reshape(xe_sm_grad, [-1, 1, K])
    new_grad = shaped_xe_sm * normalised
    #re_mag = tf.reduce_sum(new_grad ** 2.0, axis=1) ** 0.5
    #re_nomarlised = new_grad / tf.reshape(re_mag, [-1, 1, K])
    return new_grad

@tf.RegisterGradient("do_nothing")
def _normalised(unused_op, grad):
    global xe_grad
    xe_grad = grad
    return grad


class RBF:

    def __init__(self, conf):
        self.num_class = conf.num_class
        self.n = conf.n
        self.d = conf.d
        train_centres_taus = conf.train_centres_taus
        self.z = tf.get_variable("z", shape=[self.n, self.d],
                                 initializer=tf.constant_initializer(np.array([[-5.0, -5.0], [-1.0, -1.0], [5.0, 5.0], [-5.5, -5.5], [-1.5, -1.5], [5.5, 5.5]])))
                                 #initializer=tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd))
        self.y = tf.placeholder(tf.int32, shape=[self.n], name="y")
        self.rbf_c = conf.rbf_c
        self.z_bar = tf.get_variable("z_bar", shape=[self.d, self.num_class],
                                     initializer=tf.constant_initializer(np.array([[3.0, -3.0], [3.0, -3.0]])))#tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd))
        self.tau = tf.abs(tf.get_variable("tau", shape=[self.d, self.num_class],
                                          initializer=tf.constant_initializer(np.array([[0.25, 0.25], [0.25, 0.25]]))))#initializer=tf.truncated_normal_initializer(stddev=0.5)))
        self.tau_square = tf.reshape(self.tau ** 2.0, [1, self.d, self.num_class])
        g = tf.get_default_graph()
        z_re = tf.reshape(self.z, [-1, self.d, 1])
        z_tile = tf.tile(z_re, [1, 1, self.d])
        with g.gradient_override_map({'Identity': "normalised"}):
            z_identity = tf.identity(z_tile, name='Identity')
        x_diff = tf.subtract(z_identity, tf.reshape(self.z_bar, [1, self.d, self.num_class]), name='Sub')
        self.x_diff_sq = x_diff ** 2.0
        self.weighted_x_diff_sq = tf.multiply(self.tau_square, self.x_diff_sq)
        self.neg_dist = -tf.reduce_sum(self.weighted_x_diff_sq, axis = 1)
        self.exp = tf.exp(self.neg_dist)
        self.rbf = self.rbf_c * self.exp
        with g.gradient_override_map({'Identity': "stub_and_save"}):
            rbf_identity = tf.identity(self.rbf)
        sm = tf.nn.softmax(rbf_identity)
        with g.gradient_override_map({'Identity': "do_nothing"}):
            sm = tf.identity(sm)
        self.y_hot = tf.one_hot(self.y, self.num_class)
        loss = -tf.reduce_mean(tf.reduce_sum(self.y_hot*tf.log(sm), axis=1))
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=rbf_identity)
        if not train_centres_taus:
            var_list = [self.z]
            self.train_op = conf.optimizer(learning_rate=conf.lr).minimize(loss, var_list=var_list)
        else:
            self.train_op = conf.optimizer(learning_rate=conf.lr).minimize(loss)

    def all_ops(self):
        return self.train_op, self.z, self.z_bar, self.tau, tf.nn.softmax(self.rbf), xe_sm_grad, xe_grad # self.tau_square, self.x_diff_sq, self.weighted_x_diff_sq, self.neg_dist, self.exp



