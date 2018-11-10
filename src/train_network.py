import sys

import numpy as np
import tensorflow as tf


def train(conf, network_runner, data_set, collector):
    """Train a network the specified data_set using the parameters specified by conf.

    Args:
        conf: Specifies the way in which the network is trained.
        network_runner: A NetworkRunner used to encapsulate a network and the tensorflow session used to operate it.
        data_set: The DataSet to train the network on.
        collector: An object that collects information about the state of the network as it trains.

    Returns: An instance of NetworkRunner, an encapsulation of the trained network for easy reporting.
    """
    x = data_set.train.x
    y = data_set.train.y
    x_validation = data_set.validation.x
    y_validation = data_set.validation.y

    network = network_runner.network
    train_op = network.train_op
    all_ops = network.get_tensors()
    saver = tf.train.Saver(tf.global_variables())

    # Train
    n = data_set.train.n
    n_val = data_set.validation.n
    training_indices = np.arange(n)
    validation_indices = np.arange(n_val)
    epochs = conf.epochs
    accuracy_ss = conf.accuracy_ss
    lr = conf.lr
    for e in xrange(epochs):
        if e+1 in conf.decrease_lr_points:
            lr *= conf.decrease_lr_factor
        np.random.shuffle(training_indices)
        if conf.debug_ops:
            network_runner.feed_and_run(x, y, all_ops, indices=training_indices, lr=lr)
        else:
            network_runner.feed_and_run(x, y, train_op, indices=training_indices, lr=lr)
        print 'Epoch: '+str(e)
        network_runner.report_accuracy('Train', x, y, training_indices, accuracy_ss)
        network_runner.report_accuracy('Validation', x_validation, y_validation, validation_indices, accuracy_ss)
        print ''

        # Collect info about the network in it's current state, before it changes due to the next epochs training.
        collector.collect(network_runner, x, y)

    if network.model_save_dir:
        saver.save(network_runner.sess, network.model_save_dir+'/model.ckpt')
    return collector.results()


def load_pre_trained(network_runner):
    """Load a pre-trained neural network from it's model save directory, if it exists.

    Args:
        network_runner: A NetworkRunner used to encapsulate a network and the tensorflow session used to operate it.
    """
    sess = network_runner.sess
    network = network_runner.network
    saver = tf.train.Saver(tf.global_variables())

    latest = tf.train.latest_checkpoint(network.model_save_dir)
    if not latest:
        print "No checkpoint to continue from in", network.model_save_dir
        sys.exit(1)
    print "resume", latest
    saver.restore(sess, latest)