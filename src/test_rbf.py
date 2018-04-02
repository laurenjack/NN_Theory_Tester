import tensorflow as tf

class RBF:

    def __init__(self, conf):
        self.num_class = conf.num_class
        self.n = conf.n
        self.d = conf.d
        self.z = tf.get_variable("z", shape=[self.n, self.d],
                                 initializer=tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd))
        self.y = tf.placeholder(tf.int32, shape=[self.n], name="y")
        self.rbf_c = conf.rbf_c
        self.z_bar = tf.get_variable("z_bar", shape=[self.d, self.num_class],
                                     initializer=tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd))
        self.tau = tf.get_variable("tau", shape=[self.d, self.num_class],
                                     initializer=tf.truncated_normal_initializer(stddev=1.0))
        self.tau_square = tf.reshape(self.tau ** 2.0, [1, self.d, self.num_class])
        self.x_diff_sq = (tf.reshape(self.z, [-1, self.d, 1]) - tf.reshape(self.z_bar, [1, self.d, self.num_class])) ** 2.0
        self.weighted_x_diff_sq = tf.multiply(self.tau_square, self.x_diff_sq)
        self.neg_dist = -tf.reduce_sum(self.weighted_x_diff_sq, axis = 1)
        self.exp = tf.exp(self.neg_dist)
        rbf = self.rbf_c * self.exp
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=rbf)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=conf.lr).minimize(loss)

    def all_ops(self):
        return self.train_op, self.z, self.z_bar, self.tau, # self.tau_square, self.x_diff_sq, self.weighted_x_diff_sq, self.neg_dist, self.exp



