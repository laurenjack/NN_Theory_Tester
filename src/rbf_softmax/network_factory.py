import os

import numpy as np
import tensorflow as tf

import src.data_set as ds
import src.feed_forward_network
import src.network_runner as nr
import src.rbf_softmax.rbf
import src.resnet
import src.train_network
import src.vanilla_softmax
from src.rbf_softmax import collector as coll, configuration
from src.synthetic_experiments import artificial_problem as ap

_FEED_FORWARD = 'feed_forward'
_RESNET = 'resnet'
_RBF_STORE = 'rbf'
_VANILLA_STORE = 'plain'


def create_and_train_network(conf):
    """Create and train a neural network for an rbf-softmax experiment.

    This method should only be used in the context of an rbf softmax experiment. It is responsible for handling the
    complexity of an configuration.RbfSoftmaxConfiguration instance. It will use the passed in configuration to
    construct a network, train it, and return it encapsulated in an instance of NetworkRunner

    Args:
        conf: see configuration.RbfSoftmaxConfiguration

    Returns:
        A NetworkRunner. This network runner encapsulates the network as specified by conf, and is fully trained as
        specified by conf
    """

    # It's worth viewing the documentation/comments in configuration.RbfSoftmaxConfiguration before reading or
    # modifying this function.
    graph = tf.Graph()
    with graph.as_default():
        data_set = _load_data_set(conf)
        network = _build_network(conf, src.data_set, 'network0')
        network_runner = nr.build_network_runner(graph, network, conf.m, conf.is_rbf)

        if conf.is_rbf:
            collector = coll.build_rbf_collector(src.data_set, conf.animation_ss)
        else:
            collector = coll.NullCollector()

        training_results = _train_or_load(conf, src.network_runner, src.data_set, collector)
    return src.network_runner, src.data_set, training_results


def create_and_train_n_networks(conf):
    """Create and train n neural networks for an rbf-softmax experiment, where n is specified by conf.n_networks.

        Useful for testing the transerability of adverserial examples. This method should only be used in the context of
        an rbf softmax experiment. It is responsible for handling the complexity of an
        configuration.RbfSoftmaxConfiguration instance. It will use the passed in configuration to construct a list of
        n networks with identicial architectures yet different inital variables. Then it will train each using the same
        data set and hyper-parameters, returning a lsit of network runners for each network.

        Args:
            conf: see configuration.RbfSoftmaxConfiguration.

        Returns:
            A list of NetworkRunners. These network runners encapsulate the networks as specified by conf, and are fully
            trained as specified by conf.
        """

    # It's worth viewing the documentation/comments in configuration.RbfSoftmaxConfiguration before reading or
    # modifying this function.
    data_set = _load_data_set(conf)
    network_runners = []
    for i in xrange(conf.n_networks):
        graph = tf.Graph()
        with graph.as_default():
            network_id = 'network{}'.format(i)
            # Model save directory disabled in multi-network case.
            network = _build_network(conf, src.data_set, network_id)
            network_runner = nr.build_network_runner(graph, network, conf.m, conf.is_rbf)
            collector = coll.NullCollector()
            _train_or_load(conf, src.network_runner, src.data_set, collector)
            network_runners.append(src.network_runner)
    return network_runners, src.data_set



def _load_data_set(conf):
    if conf.is_resnet:
        # TODO(Jack) Replace with proper dataset constants
        if conf.dataset_name == 'bird_or_bicycle':
            data_set = ds.load_bird_or_bicycle()
        else:
            data_set = ds.load_cifar(conf.data_dir, conf.just_these_classes)
        configuration.validate(conf, src.data_set)
    else:
        # Use an artificial data set
        if conf.is_artificial_data:
            # TODO(Jack) this is yuck refactor
            data_set = ap.simple_identical_plane(conf.n // conf.num_class, conf.artificial_in_dim, conf.num_class)
        # Use MNIST
        else:
            data_set = ds.load_mnist()
    return src.data_set


def _build_network(conf, data_set, network_id):
    # Specify a place to store/load the model
    model_save_dir = conf.model_save_dir
    if model_save_dir:
        net_dir = _FEED_FORWARD
        if conf.is_resnet:
            net_dir = _RESNET
        end_dir = _VANILLA_STORE
        if conf.is_rbf:
            end_dir = _RBF_STORE
        model_save_dir = os.path.join(conf.model_save_dir, net_dir, end_dir, network_id)
        # TODO(Jack) Replace with proper dataset constants
        if conf.is_resnet and conf.dataset_name == 'bird_or_bicycle':
            model_save_dir = os.path.join(model_save_dir, 'bird_or_bicycle')

    end = _build_network_end(conf, src.data_set, network_id)
    # Create a resnet
    if conf.is_resnet:
        network = src.resnet.Resnet(conf, end, model_save_dir, src.data_set.image_width, src.data_set.image_crop_size)
    # Create a feed forward network
    else:
        num_inputs = src.data_set.train.x.shape[1]
        network = src.feed_forward_network.FeedForward(conf, end, model_save_dir, num_inputs)
    return network


def _build_network_end(conf, data_set, network_id):
    num_class = src.data_set.num_class
    if conf.is_rbf:
        z_bar_init = tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd)
        tau_init = tf.constant_initializer(conf.tau_init * np.ones(shape=[conf.d, num_class]))
        return src.rbf_softmax.rbf.Rbf(conf, z_bar_init, tau_init, num_class, network_id)
    return src.vanilla_softmax.VanillaSoftmax(num_class, network_id)


def _train_or_load(conf, network_runner, data_set, collector):
    if not conf.do_train:
        src.train_network.load_pre_trained(src.network_runner)
        return None
    return src.train_network.train(conf, src.network_runner, src.data_set, collector)

