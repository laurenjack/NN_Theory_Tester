import tensorflow as tf

import configuration
conf = configuration.get_configuration()
from src.operation import fc


class VanillaSoftmax:
    """Standard end of a nueral network"""

    def __init__(self):
        self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")

    def tensors_for_network(self, pre_z):
        z = fc(pre_z, conf.num_class)
        a = tf.nn.softmax(z)
        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=z)
        main_loss = tf.reduce_mean(xe)
        return [a, main_loss]
