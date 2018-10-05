import numpy as np
import tensorflow as tf

import feed_forward_network
import resnet
import data_set
import train_network
import rbf
from vanilla_softmax import VanillaSoftmax
from artificial_problem import simple_identical_plane


_RBF_STORE = 'resnet_rbf'
_VANILLA_STORE = 'resnet_plain'


def create_and_train_network(conf):
    """Create and train a neural network for the rbf-softmax experiment.

    Args:
        conf: see configuration.RbfSoftmaxConfiguration

    Returns:
        A NetworkRunner. This network runner encapsulates the network was specified by conf, and is fully trained as
        specified by conf
    """
    graph = tf.Graph()
    with graph.as_default():
        if conf.is_rbf:
            z_bar_init = tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd)
            tau_init = tf.constant_initializer(0.5 * np.ones(shape=[conf.d, conf.num_class])) # / float(conf.d) ** 0.5
            end = rbf.RBF(z_bar_init, tau_init)
        else:
            end = VanillaSoftmax()
        if conf.is_resnet:
            if conf.is_rbf:
                model_save_dir = _append(conf.model_save_dir)
            else:
                model_save_dir = '/home/laurenjack/models/resnet_plain'
            ds = data_set.load_cifar()
            network = resnet.Resnet(end, model_save_dir)
        else:
            if conf.is_artificial_data:
                ds = simple_identical_plane(conf.n // conf.num_class, conf.artificial_in_dim, conf.num_class)
            else:
                ds = data_set.load_mnist()
            num_inputs = ds.X_train.shape[1]
            network = feed_forward_network.Network(num_inputs, end)
        if not conf.do_train and conf.is_resenet:
            network_runner = train_network.load_pre_trained(graph, network)
        else:
            network_runner = train_network.train(graph, network, ds)
    return network_runner, ds


def _append(directory_name, next_dir):
    if directory_name.endswith('/'):
        return directory_name + next_dir
    return directory_name + '/' + next_dir