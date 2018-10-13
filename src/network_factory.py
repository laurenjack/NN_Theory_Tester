import os

import numpy as np
import tensorflow as tf

import feed_forward_network
import resnet
import data_set
import train_network
import rbf
import vanilla_softmax
import artificial_problem as ap


_RBF_STORE = 'resnet_rbf'
_VANILLA_STORE = 'resnet_plain'


def create_and_train_network(conf):
    """Create and train a neural network for the rbf-softmax experiment.

    This method should only be used in the context of an rbf softmax experiment. It is responsible for handling the
    complexity of an configuration.RbfSoftmaxConfiguration instance. It will use the passed in configuration to
    construct a network, train it, and return it encapsulated in an instance of NetworkRunner

    Args:
        conf: see configuration.RbfSoftmaxConfiguration

    Returns:
        A NetworkRunner. This network runner encapsulates the network was specified by conf, and is fully trained as
        specified by conf
    """
    graph = tf.Graph()
    with graph.as_default():
        # It's worth viewing the documentation/comments in configuration.RbfSoftmaxConfiguration before reading or
        # modifying this function.
        if conf.is_rbf:
            z_bar_init = tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd)
            tau_init = tf.constant_initializer(conf.tau_init * np.ones(shape=[conf.d, conf.num_class]))
            end = rbf.RBF(conf, z_bar_init, tau_init)
        else:
            end = vanilla_softmax.VanillaSoftmax(conf)
        if conf.is_resnet:
            if conf.is_rbf:
                model_save_dir = os.path.join(conf.model_save_dir, _RBF_STORE)
            else:
                model_save_dir = os.path.join(conf.model_save_dir, _VANILLA_STORE)
            ds = data_set.load_cifar(conf.data_dir)
            network = resnet.Resnet(end, model_save_dir)
        else:
            if conf.is_artificial_data:
                ds = ap.simple_identical_plane(conf.n // conf.num_class, conf.artificial_in_dim, conf.num_class)
            else:
                ds = data_set.load_mnist()
            num_inputs = ds.X_train.shape[1]
            network = feed_forward_network.Network(num_inputs, end)
        if not conf.do_train and conf.is_resenet:
            network_runner = train_network.load_pre_trained(graph, network)
        else:
            network_runner = train_network.train(graph, network, ds)
    return network_runner, ds