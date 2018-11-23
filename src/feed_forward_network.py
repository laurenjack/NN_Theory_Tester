import tensorflow as tf
import numpy as np

import rbf
import network


@tf.RegisterGradient('sigmoid_stub')
def _sigmoid_stub(unused_op, grad):
    return grad # * 3.0


class FeedForward(network.Network):
    """Represents a feed-forward neural network for classification.

    This model is a Multi-Layer-Perceptron with a vector of inputs per example (e.g. input shape = n * p).
    It is fully connected and uses ReLUs in all hidden layers. The end, e.g. softmax or rbf-softmax is specified as
    argument to the constructor.
    """

    def __init__(self, conf, end, model_save_dir, num_inputs):
        """
        Args:
            conf: A static set of properties to configure the network.
            end: The last layer/part of the network, e.g. an rbf-softmax end with a cross entropy loss function.
            model_save_dir: The directory to save all this network's variables.
            num_inputs: The number of inputs for the network, e.g. could be 784, for the 784 pixels of MNIST.
        """
        input_shape = [None, num_inputs]
        super(FeedForward, self).__init__(end, input_shape, False, model_save_dir)
        d = conf.d

        self.bs = []
        # Feed-forward
        hidden_sizes = conf.hidden_sizes
        ins = [num_inputs] + hidden_sizes
        outs = hidden_sizes + [d]
        a = self.x
        for l, inp, out in zip(range(len(outs[:-1])), ins[:-1], outs[:-1]):
            # a = self._create_layer(a, l, [inp, out], activation_func=tf.nn.relu)
            a, b = self._create_shredding_layer(a, l, [inp, out])
            self.bs.append(b)
        pre_z = self._create_layer(a, l + 1, [ins[-1], outs[-1]])
        #pre_z, b = self._create_shredding_layer(a, l+1, [ins[-1], outs[-1]])
        #self.bs.append(b)
        # TODO(Jack) deal with extra layer

        self.all_end_tensors = self.end.tensors_for_network(pre_z)
        self.a = self.all_end_tensors[0]
        self.original_loss = self.all_end_tensors[1]
        self.b_loss = 0.0
        for b in self.bs:
            self.b_loss += tf.reduce_mean(b)
        self.loss = -(100.0 - self.original_loss) * self.b_loss + self.original_loss
        self.train_op = tf.train.MomentumOptimizer(learning_rate=conf.lr, momentum=0.9).minimize(self.original_loss)

    def _create_layer(self, a, l, shape, activation_func=None):
        weights_init = tf.contrib.layers.variance_scaling_initializer()
        W = tf.get_variable('W'+str(l),
                        shape=shape,
                        initializer=weights_init)
        bias_init = tf.zeros_initializer()
        b = tf.get_variable('b'+str(l),
                            shape[1],
                            initializer=bias_init)
        # a = tf.nn.xw_plus_b(a, W, b)
        a = tf.matmul(a, W)
        if activation_func is not None:
            return activation_func(a)
        return a


    def _create_shredding_layer(self,  a, l, shape):
        """Layer designed to shred the redundant space of a neural network.
        """
        weights_init = tf.contrib.layers.variance_scaling_initializer()
        w_base = tf.get_variable('w_base' + str(l),
                            shape=shape,
                            initializer=weights_init)
        W = w_base / tf.reduce_sum(w_base ** 2.0, axis=0) ** 0.5

        graph = tf.get_default_graph()
        b_base_init = tf.constant_initializer(-3.0 * np.ones(shape=[shape[1]]))
        b_base = tf.get_variable('b_base'+str(l), shape=[shape[1]], initializer=b_base_init)
        sigmoid_name = 'Sigmoid_{}'.format(l)
        with graph.gradient_override_map({'Sigmoid': 'sigmoid_stub'}):
            b = tf.nn.sigmoid(b_base, name='Sigmoid')  # , name='Sigmoid'

        a_magnitude = tf.reduce_sum(a ** 2.0, axis=1) ** 0.5
        outer_product = tf.reshape(a_magnitude, [-1, 1]) * tf.reshape(b, [1, shape[1]])
        a = 1.01 * tf.matmul(a, W) - outer_product
        a = 1.00 / (1.01- b) ** 2.0 * tf.nn.relu(a)
        return a, b



