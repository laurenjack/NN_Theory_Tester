import tensorflow as tf
import numpy as np

class RBF:

    def __init__(self, conf):
        self.num_class = conf.num_class
        self.n = conf.n
        self.d = conf.d
        train_centres_taus = conf.train_centres_taus
        self.z = tf.get_variable("z", shape=[self.n, self.d],
                                 initializer=tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd))
        self.y = tf.placeholder(tf.int32, shape=[self.n], name="y")
        self.rbf_c = conf.rbf_c
        self.z_bar = tf.get_variable("z_bar", shape=[self.d, self.num_class],
                                     initializer=tf.constant_initializer(np.array([[3.0, -3.0], [3.0, -3.0]])))#tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd))
        self.tau = tf.abs(tf.get_variable("tau", shape=[self.d, self.num_class],
                                          initializer=tf.constant_initializer(np.array([[0.25, 0.25], [0.25, 0.25]]))))#initializer=tf.truncated_normal_initializer(stddev=0.5)))
        self.tau_square = tf.reshape(self.tau ** 2.0, [1, self.d, self.num_class])
        self.x_diff_sq = (tf.reshape(self.z, [-1, self.d, 1]) - tf.reshape(self.z_bar, [1, self.d, self.num_class])) ** 2.0
        self.weighted_x_diff_sq = tf.multiply(self.tau_square, self.x_diff_sq)
        self.neg_dist = -tf.reduce_sum(self.weighted_x_diff_sq, axis = 1)
        self.exp = tf.exp(self.neg_dist)
        rbf = self.rbf_c * self.exp
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=rbf)
        if not train_centres_taus:
            var_list = [self.z]
            self.train_op = conf.optimizer(learning_rate=conf.lr).minimize(loss, var_list=var_list)
        else:
            self.train_op = conf.optimizer(learning_rate=conf.lr).minimize(loss)

    def all_ops(self):
        return self.train_op, self.z, self.z_bar, self.tau, # self.tau_square, self.x_diff_sq, self.weighted_x_diff_sq, self.neg_dist, self.exp



