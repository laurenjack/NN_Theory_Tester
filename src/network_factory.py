import os

import numpy as np
import tensorflow as tf

import configuration
import feed_forward_network
import resnet
import data_set as ds
import train_network
import rbf
import vanilla_softmax
import artificial_problem as ap
import collector as coll
import network_runner as nr


_FEED_FORWARD = 'feed_forward'
_RESNET = 'resnet'
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

    # It's worth viewing the documentation/comments in configuration.RbfSoftmaxConfiguration before reading or
    # modifying this function.
    graph = tf.Graph()
    with graph.as_default():
        # Create a resnet
        if conf.is_resnet:
            # Specify a place to store/load the model
            if conf.model_save_dir:
                net_dir = _FEED_FORWARD
                if conf.is_resnet:
                    net_dir = _RESNET
                end_dir = _VANILLA_STORE
                if conf.is_rbf:
                    end_dir = _RBF_STORE
                model_save_dir = os.path.join(conf.model_save_dir, net_dir, end_dir)

            data_set = ds.load_cifar(conf.data_dir, conf.just_these_classes)
            configuration.validate(conf, data_set)
            end = _build_network_end(conf, data_set)
            network = resnet.Resnet(conf, end, model_save_dir, data_set.image_width)
        # Create a feed forward netowrk
        else:
            # Use an artificial data set
            if conf.is_artificial_data:
                # TODO(Jack) this is yuck refactor
                data_set = ap.simple_identical_plane(conf.n // conf.num_class, conf.artificial_in_dim, conf.num_class)
            # Use MNIST
            else:
                data_set = ds.load_mnist()

            num_inputs = data_set.train.x.shape[1]
            end = _build_network_end(conf, data_set)
            network = feed_forward_network.FeedForward(conf, end, num_inputs)

        network_runner = nr.build_network_runner(graph, network, conf.m, conf.is_rbf)

        # Load the network from a pre-saved model
        if not conf.do_train and conf.is_resenet:
            train_network.load_pre_trained(network_runner)
        # Train the network
        else:
            if conf.is_rbf:
                collector = coll.build_rbf_collector(data_set, conf.animation_ss)
            else:
                collector = coll.NullCollector()
            traing_results = train_network.train(conf, network_runner, data_set, collector)
    return network_runner, data_set, traing_results


def _build_network_end(conf, data_set):
    num_class = data_set.num_class
    if conf.is_rbf:
        z_bar_init = tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd)
        tau_init = tf.constant_initializer(conf.tau_init * np.ones(shape=[conf.d, num_class]))
        return rbf.Rbf(conf, z_bar_init, tau_init, num_class)
    return vanilla_softmax.VanillaSoftmax(num_class)
