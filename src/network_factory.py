import numpy as np
import tensorflow as tf
import configuration
conf = configuration.get_conf()
from feed_forward_network import Network
from resnet import Resnet
from data_set import load_mnist
from data_set import load_cifar
import train_network
import rbf
from vanilla_softmax import VanillaSoftmax


def create_and_train_network():
    graph = tf.Graph()
    with graph.as_default():
        tf.placeholder(tf.int32, shape=[], name="batch_size")
        if conf.is_rbf:
            z_bar_init = tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd)
            tau_init = tf.constant_initializer(0.5 / float(conf.d) ** 0.5 * np.ones(shape=[conf.d, conf.num_class]))
            end = rbf.RBF(z_bar_init, tau_init)
        else:
            end = VanillaSoftmax()
        if conf.is_resnet:
            if conf.is_rbf:
                model_save_dir = '/home/laurenjack/models/resnet_rbf'
            else:
                model_save_dir = '/home/laurenjack/models/resnet_plain'
            data_set = load_cifar(conf)
            network = Resnet(conf, end, model_save_dir)
        else:

            data_set = load_mnist(conf)
            network = Network(end, conf)
        if conf.do_train:
            network_runner = train_network.train(graph, network, data_set, conf)
        else:
            network_runner = train_network.load_pre_trained(graph, network, conf)
    return network_runner, data_set
