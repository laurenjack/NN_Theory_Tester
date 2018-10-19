import sys

import numpy as np
import tensorflow as tf
from network_runner import NetworkRunner

import configuration
conf = configuration.get_configuration()
import rbf


def train(graph, network, data_set):
    num_class = data_set.num_class
    X = data_set.X_train
    Y = data_set.Y_train
    X_val = data_set.X_val
    Y_val = data_set.Y_val

    # Load the session and ops
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    train_op = network.train_op
    all_ops = network.get_ops()
    saver = tf.train.Saver(tf.global_variables())
    network_runner = NetworkRunner(network, sess, graph)

    class_wise_z_list = []
    for k in xrange(num_class):
        class_wise_z_list.append([])
    z_bars = []
    taus = []

    # Train
    n = data_set.n_train
    m = conf.m
    batch_indicies = np.arange(n)
    animation_indicies = np.random.choice(batch_indicies, size=conf.animation_ss, replace=False)
    val_batch_indicies = np.arange(X_val.shape[0])
    epochs = conf.epochs
    accuracy_ss = conf.accuracy_ss
    for e in xrange(epochs):
        if e+1 in conf.decrease_lr_points:
            conf.lr *= conf.decrease_lr_factor
        np.random.shuffle(batch_indicies)
        for k in xrange(0, n, m):
            batch = batch_indicies[k:k + m]
            if conf.debug_ops:
                op_results = network_runner.feed_and_run(X, Y, all_ops, batch, is_training=True)
            else:
                network_runner.feed_and_run(X, Y, train_op, batch, is_training=True)
        print 'Epoch: '+str(e)
        network_runner.report_accuracy('Train', batch_indicies, X, Y, accuracy_ss)
        network_runner.report_accuracy('Validation', val_batch_indicies, X_val, Y_val, accuracy_ss)
        print ''

        #At the end of epoch, calculate what will be shown for animation, (if required)
        if conf.show_animation and conf.is_rbf:
            z, z_bar, tau = network_runner.feed_and_run(X, Y, network.rbf_params(), animation_indicies, is_training=True)
            for k in xrange(num_class):
                ind_of_class = np.argwhere(Y[animation_indicies] == k)[:, 0]
                class_wise_z_list[k].append(z[ind_of_class])
            z_bars.append(z_bar)
            taus.append(tau)


    network_runner.set_ops_over_time((class_wise_z_list, z_bars, taus))

    if network.model_save_dir:
        saver.save(sess, network.model_save_dir+'/model.ckpt')
    return network_runner


def load_pre_trained(graph, network):
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(tf.global_variables())

    latest = tf.train.latest_checkpoint(network.model_save_dir)
    if not latest:
        print "No checkpoint to continue from in", network.model_save_dir
        sys.exit(1)
    print "resume", latest
    saver.restore(sess, latest)
    return NetworkRunner(network, sess, graph)