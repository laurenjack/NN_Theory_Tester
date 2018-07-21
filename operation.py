import tensorflow as tf

"""Used to for resuable op construction"""


def fc(a, num_units_out):
    num_units_in = a.get_shape()[1]
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)

    weights = _get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer)
    biases = _get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer)
    a = tf.nn.xw_plus_b(a, weights, biases)
    weight_reg = tf.nn.l2_loss(weights)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_reg)
    return a


def _get_variable(name, shape, initializer, dtype=tf.float32, trainable=True):
    """A little wrapper around tf.get_variable to do weight decay and add to resnet collection"""
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype,
                           collections=collections, trainable=trainable)