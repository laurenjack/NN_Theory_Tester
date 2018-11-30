import tensorflow as tf


def shredding_layer():
    pass


def fc(a, num_units_out):
    """Perform an affine transformation on matrix a with num_units_out outputs.

    This represents a full connected layer in a neural network, with no activation function applied.

    Args:
        a: An [m, l] tensor, the previous layer of an NN
        num_units_out: The number of rows in the output matrix.

    Returns: An [m, num_units_out] tensor, the output of the affine transformation.
    """
    num_units_in = a.get_shape()[1]
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)
    weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer)
    biases = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer)
    a = tf.nn.xw_plus_b(a, weights, biases)
    weight_reg = tf.nn.l2_loss(weights)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_reg)
    return a


def per_filter_fc(a, d):
    """Given the last conv layer of a network a, perform a separate affine transformation a_fW + b per filter, then
    stack the outputs to give a single [m, d] tensor.

    Let a have the shape m, w, w, num_filter. Where m is the batch size, w is the filter width (and height) of
    the last layer, and num_filter is the number of filters at this layer. For each filter, this function will define
    a dense matrix W with shape d / num_filter, w ** 2.0, a bias vector with of shape d / num_filter, and will apply
    a_fW + b to each filter (where a_f is all the activation of a given filter). Then it will stack the output vectors
    to produce num_filter * d / num_filter = d outputs.

    Why do this? The standard process is to use global average pooling after the last conv layer. While this works, it
    limits the dimensionality of this layer to num_filter. In the context of detecting anomalies in the z space, if the
    dimensionality is low relative to the number of input pixels, this would (in theory) make it easier to generate
    adversaries that fit the training distribution of z. So we want a larger dimensionality post conv-layer, but don't
    want to use a large and expensive fully connected layer with w ** 2.0 * num_filter * d weights. Instead, we use
    a dense W per filter, which has w ** 2.0 * d / num_filter weights, with a total of w ** 2.0 * d parameters.
    Essentially, one might think of this as a sparse FC layer with zeros where one filter i lines up with a different
    filter j, and block matrices where i=j. The upshot of using this is we get something cheaper than a conventional
    fully connected layer, while allowing for a large dimensionality in d, with a layer that is a generalisation of
    global average pooling (if one omits the scaling by the mean, global average pooling is just a per filter sum of
    activations).

    Args:
        a: an [m, w, w, num_filter] shaped tensor, representing the last layer of conv filters
        d: The desired dimensionality of the output, d

    Returns:
        An [m, d] shaped tensor, the stacked activations of num_filter affine transformations.
    """
    dims = a.get_shape()
    num_filters = dims[3].value
    num_units_in = dims[1] * dims[2]
    outputs_per_filter = d / num_filters

    # Scale the initializer down. The fact that we stack outputs_per_filter output vectors, means the initial variance
    # of the sum of these output activations will be scaled up by outputs_per_filter.
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(1.0/outputs_per_filter)

    out = []
    # Create a weight matrix for each filter, apply filter-wise matrix multiplication
    a = tf.transpose(a, [3, 0, 1, 2])
    a = tf.reshape(a, [num_filters, -1, num_units_in.value])
    for i in xrange(num_filters):
        w = tf.get_variable('weights' + str(i), shape=[num_units_in, outputs_per_filter],
                            initializer=weights_initializer)
        weight_reg = tf.nn.l2_loss(w)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_reg)
        b = tf.get_variable('biases' + str(i), shape=[outputs_per_filter], initializer=tf.zeros_initializer)
        o = tf.nn.xw_plus_b(a[i], w, b)
        out.append(o)
    out = tf.stack(out)
    out = tf.transpose(out, [1, 0, 2])
    return tf.reshape(out, [-1, d])
