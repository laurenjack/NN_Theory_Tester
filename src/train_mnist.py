import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#import rbf
import tensorflow as tf
from network_runner import NetworkRunner
from prediction_output_writer import *

def train(network, conf):
    #Load data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    train_set = mnist.train
    val_set = mnist.validation
    X = train_set.images
    Y = train_set.labels
    X_val = val_set.images
    Y_val = val_set.labels

    #Load the session and ops
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    train_op = network.train_op
    network_runner = NetworkRunner(network, sess)

    #Train
    train_accs = []
    val_accs = []
    n = conf.n
    m = conf.m
    batch_indicies = np.arange(n)
    val_batch_indicies = np.arange(X_val.shape[0])
    epochs = conf.epochs
    accuracy_ss = conf.accuracy_ss
    for e in xrange(epochs):
        np.random.shuffle(batch_indicies)
        for k in xrange(0, n, m):
            batch = batch_indicies[k:k + m]
            network_runner.feed_and_run(X, Y, train_op, batch)
        print 'Epoch: '+str(e)
        train_acc = network_runner.report_accuracy('Train', batch_indicies, accuracy_ss, X, Y)
        val_acc = network_runner.report_accuracy('Validation', val_batch_indicies, accuracy_ss, X_val, Y_val)
        print ''
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    # Report on a sample of correct and incorrect results
    corr, incorr = network_runner.sample_correct_incorrect(10, X_val, Y_val)
    corr.show()
    incorr.show()

    # Final z_bar and tau
    write_csv(X_val, Y_val, network_runner)
    z_bar, tau = sess.run((network.z_bar, network.tau))
    return z_bar, tau