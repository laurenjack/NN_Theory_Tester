import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import rbf
import tensorflow as tf

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
            _feed_and_run(batch, X, Y, train_op, network, sess)
        print 'Epoch: '+str(e)
        train_acc = _report_accuracy('Train', batch_indicies, accuracy_ss, X, Y, network, sess)
        val_acc = _report_accuracy('Validation', val_batch_indicies, accuracy_ss, X_val, Y_val, network, sess)
        print ''
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    return train_accs, val_accs


def _feed_and_run(batch, X, Y, op, network, sess):
    x = X[batch]
    y = Y[batch]
    batch_size = batch.shape[0]
    feed_dict = {network.x: x, network.rbf.y: y, rbf.batch_size: batch_size}
    z, result = sess.run((network.z, op), feed_dict=feed_dict)
    #print z
    return result


def _report_accuracy(set_name, batch_indicies, accuracy_ss, X, Y, network, sess):
    acc_batch = _random_batch(batch_indicies, accuracy_ss)
    a = _feed_and_run(acc_batch, X, Y, network.a, network, sess)
    y = Y[acc_batch]
    acc = _compute_accuracy(a, y)
    print set_name+" Accuracy: "+str(acc)
    return acc


def _random_batch(batch_indicies, m):
    return np.random.choice(batch_indicies, size=m, replace=False)


def _compute_accuracy(a, y):
    ss = y.shape[0]
    prediction = np.argmax(a, axis=1)
    correct_indicator = np.equal(prediction, y).astype(np.int32)
    return float(np.sum(correct_indicator)) / float(ss)