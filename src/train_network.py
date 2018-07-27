import numpy as np
import tensorflow as tf
from network_runner import NetworkRunner
import sys


def train(graph, network, data_set, conf):
    X = data_set.X_train
    Y = data_set.Y_train
    X_val = data_set.X_val
    Y_val = data_set.Y_val

    # Load the session and ops
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    train_op = network.train_op
    saver = tf.train.Saver(tf.global_variables())
    network_runner = NetworkRunner(graph, network, sess, conf)

    # Train
    n = conf.n
    m = conf.m
    batch_indicies = np.arange(n)
    val_batch_indicies = np.arange(X_val.shape[0])
    epochs = conf.epochs
    accuracy_ss = conf.accuracy_ss
    for e in xrange(epochs):
        if e+1 in conf.decrease_lr_points:
            conf.lr *= conf.decrease_lr_factor
        np.random.shuffle(batch_indicies)
        for k in xrange(0, n, m):
            batch = batch_indicies[k:k + m]
            _, a, z, z_bar, tau = network_runner.feed_and_run(X, Y, [train_op, network.a, network.z, network.z_bar, network.tau], batch, is_training=True)
            # network_runner.feed_and_run(X, Y, train_op, batch, is_training=True)
        print 'Epoch: '+str(e)
        network_runner.report_accuracy('Train', batch_indicies, accuracy_ss, X, Y)
        network_runner.report_accuracy('Validation', val_batch_indicies, accuracy_ss, X_val, Y_val)
        print ''

    if network.model_save_dir is not None:
        saver.save(sess, network.model_save_dir+'/model.ckpt')
    return network_runner


def load_pre_trained(graph, network, conf):
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(tf.global_variables())

    latest = tf.train.latest_checkpoint(network.model_save_dir)
    if not latest:
        print "No checkpoint to continue from in", network.model_save_dir
        sys.exit(1)
    print "resume", latest
    saver.restore(sess, latest)
    return NetworkRunner(graph, network, sess, conf)