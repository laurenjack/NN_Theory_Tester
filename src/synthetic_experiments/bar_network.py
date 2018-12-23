import tensorflow as tf
import src.operation as op

HEIGHT = 2
WIDTH = 2

class BarNetwork:

    def __init__(self, vc, num_filters, lr):
        self.vc = vc
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH])
        self.y = self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training') # TODO remove once refactored
        # Two convulotional layers
        self.a1 = self._custom_conv_layer(self.x, num_filters[0], 1)
        #self.a2 = self._custom_conv_layer(self.a1, num_filters[1], 2)

        # Global average pooling, i.e average of each feature
        ap = tf.reduce_mean(self.a1, axis=2, name="avg_pool")
        self.z, self.fc_weight = op.fc(ap, 2)

        # Softmax
        self.a = tf.nn.softmax(self.z)
        self.xe = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.y, 2), logits=self.z)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(self.xe)


    def _custom_conv_layer(self, input, num_filter, l):
        in_shape = input.get_shape()
        height = in_shape[1].value
        width = in_shape[2].value
        # Create the weights and biases for this layer
        weight_init = tf.contrib.layers.variance_scaling_initializer()
        bias_init = tf.ones_initializer()
        wb = [(self.vc.get_variable('W'+str(l)+str(i), shape=[height, 1], initializer=weight_init),
               self.vc.get_variable('b'+str(l)+str(i), shape=[], initializer=bias_init)) for i in xrange(num_filter)]
        w, b = zip(*wb)
        self.b = b
        self.w = w

        # Pass each filter over each stack of pixels
        a = []
        for i in xrange(width):
            a_i = []
            for j in xrange(num_filter):
                wj = tf.reshape(w[j], shape=[-1, 1])
                in_i = input[:, :, i]
                z_ij = tf.matmul(in_i, wj) # + b[j]
                a_ij = tf.nn.relu(z_ij)
                a_i.append(a_ij)
            a_i = tf.concat(a_i, axis=1)
            a.append(tf.reshape(a_i, shape=[-1, num_filter, 1]))
        a = tf.concat(a, axis=2)
        return a