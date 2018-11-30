import tensorflow as tf
import numpy as np

import network


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

        # Create random orthogonality filter
        orthogonality_filter = 0.03 + abs(np.random.randn(784))
        orthogonality_filter = tf.Variable(orthogonality_filter, trainable=False, dtype=tf.float32)
        # one_indices = np.arange(784)
        # np.random.shuffle(one_indices)
        # step = 16
        # for i in xrange(0, 784, step):
        #     current_one_indices = one_indices[i:i+step]
        #     orthogonality_filter[i:i+step, current_one_indices] = 1.0
        # print np.sum(orthogonality_filter, axis=1)
        # orthogonality_filter = tf.constant(16 * orthogonality_filter)
        # orthogonality_filter = tf.transpose(orthogonality_filter)

        # Feed-forward
        hidden_sizes = conf.hidden_sizes
        ins = [num_inputs] + hidden_sizes
        outs = hidden_sizes + [d]
        a = self.x * orthogonality_filter
        for l, inp, out in zip(range(len(outs[:-1])), ins[:-1], outs[:-1]):
            a = self._create_layer(a, l, [inp, out], activation_func=tf.nn.relu)
        orthogonality_filter = 0.03 + abs(np.random.randn(784))
        # orthogonality_filter = orthogonality_filter / np.sum(orthogonality_filter ** 2.0) ** 0.5
        orthogonality_filter = tf.Variable(orthogonality_filter, trainable=False, dtype=tf.float32)
        a = a * orthogonality_filter
        pre_z = self._create_layer(a, l + 1, [ins[-1], outs[-1]])
        # TODO(Jack) deal with extra layer

        self.all_end_tensors = self.end.tensors_for_network(pre_z)
        self.a = self.all_end_tensors[0]
        self.loss = self.all_end_tensors[1]
        self.train_op = tf.train.MomentumOptimizer(learning_rate=conf.lr, momentum=0.9).minimize(self.loss)

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



