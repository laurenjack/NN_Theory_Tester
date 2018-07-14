import numpy as np
import tensorflow as tf
from network_runner import NetworkRunner


def train(graph, network, data_set, conf):
    X = data_set.X_train
    Y = data_set.Y_train
    X_val = data_set.X_val
    Y_val = data_set.Y_val

    # Load the session and ops
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    train_op = network.train_op
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
            network_runner.feed_and_run(X, Y, train_op, batch, is_training=True)
        print 'Epoch: '+str(e)
        network_runner.report_accuracy('Train', batch_indicies, accuracy_ss, X, Y)
        network_runner.report_accuracy('Validation', val_batch_indicies, accuracy_ss, X_val, Y_val)
        print ''

    return network_runner
