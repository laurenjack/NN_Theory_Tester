import tensorflow as tf
import operation


class VanillaSoftmax:
    """Standard softmax end of a neural network
    """

    def __init__(self, num_class):
        self.num_class = num_class
        self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")

    def tensors_for_network(self, pre_z):
        """Build a standard softmax ending to a Neural network, with a cross entropy loss function.

        Args:
            pre_z: An m x d floating point tensor, which represents the current batch at the last layer of the
            neural network before the softmax

        Returns:
            A list of tensors, for use in the construction of a neural network.
        """
        z = operation.fc(pre_z, self.num_class)
        a = tf.nn.softmax(z)
        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=z)
        main_loss = tf.reduce_mean(xe)
        return [a, main_loss]
