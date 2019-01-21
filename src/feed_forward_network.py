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
            conf: A static set of properties to configure the network
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
        pre_z = self._create_layer(a, l+1, [ins[-1], outs[-1]], activation_func=tf.nn.relu)

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
        a = tf.nn.xw_plus_b(a, W, b)
        # Add the activations from each layer to a list for post training reporting
        self.activation_list.append(a)
        if activation_func is not None:
            return activation_func(a)
        return a


