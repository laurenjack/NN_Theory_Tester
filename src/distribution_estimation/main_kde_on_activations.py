import tensorflow as tf
import numpy as np

import network_configuration
from src import simple_convolution as sc
from src.rbf_softmax import network_factory
from src import data_set as ds


def train_network(conf, mnist, tensors, saver):
    lr, x, y, train_op, activations, accuracy = tensors
    m = conf.m
    lr_value = conf.lr
    accuracy_ss = conf.accuracy_ss
    train = mnist.train
    validation = mnist.validation
    n = train.n
    n_val = validation.n
    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    indices = np.arange(n)
    indices_val = np.arange(n_val)
    for e in xrange(conf.epochs):
        e = e+1
        np.random.shuffle(indices)
        for k in xrange(0, n, m):
            batch = indices[k:k+m]
            session.run(train_op, feed_dict={lr: lr_value, x: train.x[batch], y: train.y[batch]})
        np.random.shuffle(indices_val)
        sample = indices_val[0:accuracy_ss]
        acc_train = session.run(accuracy, feed_dict={x: train.x[indices[0: accuracy_ss]],
                                                     y: train.y[indices[0: accuracy_ss]]})
        acc_val = session.run(accuracy, feed_dict={x: validation.x[sample], y: validation.y[sample]})
        print 'Epoch {}: {} {}'.format(e, acc_train, acc_val)
    saver.save(session, conf.model_save_dir + '/model.ckpt')


def run(conf):
    mnist = ds.load_mnist()
    # Create a neural network
    simple_convolution = sc.SimpleConvolutionService(mnist.image_width)
    tensors = simple_convolution.get_tensors()
    saver = tf.train.Saver(tf.global_variables())
    train_network(conf, mnist, tensors, saver)




if __name__ == '__main__':

    class Conf(object):

        def __init__(self):
            self.m = 32
            self.lr = 0.03
            self.epochs = 10
            self.accuracy_ss = 100
            self.model_save_dir = '/Users/jack/models/simple_conv'

    conf = Conf()
    run(conf)

    # conf_network = network_configuration.get_configuration()
    # adversarial_ss = conf_network.adversarial_ss
    # class_to_adversary = conf_network.class_to_adversary_class
    # network_runner, data_set, training_results = network_factory.create_and_train_network(conf_network)