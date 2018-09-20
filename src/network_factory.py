import numpy as np
import tensorflow as tf
from configuration import conf
from feed_forward_network import Network
from resnet import Resnet
from data_set import load_mnist
from data_set import load_cifar
import train_network
import rbf
from vanilla_softmax import VanillaSoftmax
from artificial_problem import simple_identical_plane


def create_and_train_network():
    graph = tf.Graph()
    with graph.as_default():
        if conf.is_rbf:
            z_bar_init = tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd)
            tau_init = tf.constant_initializer(0.5  * np.ones(shape=[conf.d, conf.num_class])) # / float(conf.d) ** 0.5
            end = rbf.RBF(z_bar_init, tau_init)
        else:
            end = VanillaSoftmax()
        if conf.is_resnet:
            if conf.is_rbf:
                model_save_dir = '/home/laurenjack/models/resnet_rbf'
            else:
                model_save_dir = '/home/laurenjack/models/resnet_plain'
            data_set = load_cifar()
            network = Resnet(end, model_save_dir)
        else:
            if conf.is_artificial_data:
                data_set = simple_identical_plane(conf.n // conf.num_class, conf.artificial_in_dim, conf.num_class)
            else:
                data_set = load_mnist()
            num_inputs = data_set.X_train.shape[1]
            network = Network(num_inputs, end)
        if conf.do_train:
            network_runner = train_network.train(graph, network, data_set)
        else:
            network_runner = train_network.load_pre_trained(graph, network)
    return network_runner, data_set
