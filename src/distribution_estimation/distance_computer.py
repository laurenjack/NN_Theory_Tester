import tensorflow as tf


def network_distance(x, x_correct, network_runner):
    """
    Compute the network distance between an unkown batch of image x, against a set of correct validation example
    x_correct, for the network. For a definition of network distance, see the following for a description of network
    distance:
    """
    network = network_runner.network
    with network_runner.graph.as_default():
        m, d = x.shape
        r = x_correct.shape[0]
        inner_weight_list = network.weight_list
        activation_list = network.activation_list
        softmax_weight = network.all_end_tensors[3]
        all_weights = inner_weight_list + [softmax_weight]
        # Compute the distance weight matrix
        distance_weight = inner_weight_list[0]
        for W in all_weights[1:]:
            distance_weight = tf.matmul(distance_weight, W)
        distance_weight = tf.reduce_sum(abs(distance_weight), axis=1)
        # Find weighted difference between examples.
        difference = tf.reshape(x, [m, 1, d]) - tf.reshape(x_correct, [1, r, d])
        distance = tf.reduce_sum(abs(difference) * distance_weight, axis=2)
    return distance
