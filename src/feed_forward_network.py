import tensorflow as tf

import rbf
import network


class FeedForward(network.Network):
    """Represents a feed-forward neural network for classification.

    This model is a Multi-Layer-Perceptron with a vector of inputs per example (e.g. input shape = n * p).
    It is fully connected and uses ReLUs in all hidden layers. The end, e.g. softmax or rbf-softmax is specified as
    argument to the constructor.
    """

    def __init__(self, conf, end, num_inputs):
        """
        Args:
            conf: see configuration.RbfSoftmaxConfiguration
            end: The last layer/part of the network, e.g. an rbf-softmax end with a cross entropy loss function
            num_inputs: The number of inputs for the network, e.g. could be 784, for the 784 pixels of MNIST.
        """
        input_shape = [None, num_inputs]
        super(FeedForward, self).__init__(end, input_shape, False)
        d = conf.d

        # Feed-forward
        hidden_sizes = conf.hidden_sizes
        ins = [num_inputs] + hidden_sizes
        outs = hidden_sizes + [d]
        a = self.x
        for l, inp, out in zip(range(len(outs[:-1])), ins[:-1], outs[:-1]):
            a = self._create_layer(a, l, [inp, out], activation_func=tf.nn.relu)
        pre_z = self._create_layer(a, l+1, [ins[-1], outs[-1]])

        self.all_end_ops = self.end.tensors_for_network(pre_z)
        self.a = self.all_end_ops[0]
        self.loss = self.all_end_ops[1]
        self.train_op = conf.optimizer(learning_rate=conf.lr, momentum=0.9).minimize(self.loss)

    def get_ops(self):
        return [self.train_op] + self.all_end_ops

    def _create_layer(self, a, l, shape, activation_func=None):
        weights_init = tf.contrib.layers.variance_scaling_initializer()
        W = tf.get_variable('W'+str(l),
                        shape=shape,
                        initializer=weights_init)
        bias_init = tf.zeros_initializer()
        b = tf.get_variable('b'+str(l),
                            shape[1],
                            initializer=bias_init)
        a = tf.nn.xw_plus_b(a, W, b)
        if activation_func is not None:
            return activation_func(a)
        return a

    # TODO(Jack) cull from here
    def get_x(self):
        return self.x

    def get_y(self):
        return self.end.y

    def get_y_hot(self):
        return self.end.y_hot

    def get_batch_size(self):
        return self.end.batch_size

    def get_lr(self):
        return self.lr

    # TODO(Jack) to here


    def has_rbf(self):
        return isinstance(self.end, rbf.Rbf)



