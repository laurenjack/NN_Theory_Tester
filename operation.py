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


def per_filter_fc(a):
    dims = a.get_shape()
    num_filters = dims[3]
    num_units_in = dims[1] * dims[2]

    weights_initializer = tf.contrib.layers.variance_scaling_initializer(1.0/16.0)
    out = []
    # Create a weight matrix for each filter
    # Apply filter-wise matrix multiplication
    a = tf.transpose(a, [3, 0, 1, 2])
    a = tf.reshape(a, [num_filters.value, -1, num_units_in.value])
    for i in xrange(num_filters):
        w = _get_variable('weights' + str(i),
                               shape=[num_units_in, num_units_in],
                               initializer=weights_initializer)
        weight_reg = tf.nn.l2_loss(w)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_reg)
        b = _get_variable('biases' + str(i),
                               shape=[num_units_in],
                               initializer=tf.zeros_initializer)
        o = tf.nn.xw_plus_b(a[i], w, b)
        out.append(o)
    # w = tf.stack(weights)
    # b = tf.stack(biases)
    out = tf.stack(out)
    out = tf.transpose(out, [1, 0, 2])
    return tf.reshape(out, [-1, num_filters.value * num_units_in.value])


def _get_variable(name, shape, initializer, dtype=tf.float32, trainable=True):
    """A little wrapper around tf.get_variable to do weight decay and add to resnet collection"""
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype,
                           collections=collections, trainable=trainable)