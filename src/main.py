import tensorflow as tf
from animator import *
import train_rbf
import train_network
from data_set import load_mnist
from data_set import load_cifar
from feed_forward_network import Network
from resnet import Resnet
from shortest_point_finder import find_shortest_point
from reporter import *
import rbf
import configuration


def run_network(conf):
    if conf.is_resnet:
        data_set = load_cifar(conf)
        network_runner = create_and_train_resnet(data_set)
    else:
        data_set = load_mnist()
        network_runner = create_and_train_network(data_set)
        # network_runner2 = create_and_train_network(data_set))

    # Report on the results
    report_single_network(network_runner, data_set, conf)
    # report_with_adversaries_from_second(network_runner1, network_runner2, data_set, conf)


def create_and_train_network(data_set):
    graph = tf.Graph()
    with graph.as_default():
        z_bar_init = tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd)
        tau_init = tf.constant_initializer(0.5 / float(conf.d) ** 0.5 * np.ones(shape=[conf.d, conf.num_class]))
        rbf_end = rbf.RBF(z_bar_init, tau_init)
        network = Network(rbf_end, conf)
        if conf.do_train:
            network_runner = train_network.train(graph, network, data_set, conf)
        else:
            network_runner = train_network.load_pre_trained(graph, network, conf)
    return network_runner


def create_and_train_resnet(data_set):
    graph = tf.Graph()
    with graph.as_default():
        resnet = Resnet(conf)
        if conf.do_train:
            network_runner = train_network.train(graph, resnet, data_set, conf)
        else:
            network_runner = train_network.load_pre_trained(graph, resnet, conf)
    return network_runner


def run_rbf_test(conf):
    total_correct = 0
    for i in xrange(conf.num_runs):
        train_result = train_rbf.train(conf)
        train_result.report_incorrect()
        num_correct = train_result.num_correct
        print float(num_correct) / float(conf.n) * 100
        total_correct += num_correct
        z_bar = train_result.final_z_bar
        tau = train_result.final_tau
        sp_z_list, Cs, rbfs, z_bar_pair, tau_pair = find_shortest_point(conf, z_bar, tau)
        print Cs
        print rbfs
    print ""
    tpc = float(total_correct) / float(conf.n * conf.num_runs) * 100
    print "Total Percentage Correct: " + str(tpc)

    # Take the last a and evaluate the percentage of correctly classified points
    if conf.num_runs == 1 and conf.d == 2:
        animate(train_result, conf)
        animate_spf(z_bar_pair, tau_pair, sp_z_list, conf)

if __name__ == '__main__':
    conf = configuration.get_conf()
    #run_rbf_test(conf)
    run_network(conf)



