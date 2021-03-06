import tensorflow as tf
import numpy as np


class Trainer(object):

    def __init__(self, conf, random):
        # Services
        self.random = random
        # Configuration Constants
        self.m = conf.m
        self.r = conf.r
        self.c = conf.c
        self.d = conf.d
        self.lr_init = conf.lr_init
        self.Q_init = conf.Q_init
        self.lam_inv_init = conf.lam_inv_init
        self.fit_to_underlying_pdf = conf.fit_to_underlying_pdf
        self.epochs = conf.epochs
        self.reduce_lr_epochs = conf.reduce_lr_epochs
        self.reduce_lr_factor = conf.reduce_lr_factor
        self.show_variable_during_training = conf.show_variable_during_training

    def train(self, mv_kde_service, x, actual_lamda):
        # Placeholders
        lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
        batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        a = tf.placeholder(dtype=tf.float32, shape=[None, self.d], name='a')
        a_star1 = tf.placeholder(dtype=tf.float32, shape=[self.r, self.d], name='a_star1')
        a_star2 = tf.placeholder(dtype=tf.float32, shape=[self.r, self.d], name='a_star2')
        # Construct graph
        mv_graph = mv_kde_service.create_tensors(lr, batch_size, a, a_star1, a_star2, self.Q_init, self.lam_inv_init)
        n = x.shape[0]
        # Initialise variables
        session = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        lr_val = self.lr_init
        # Compute the step (total batch size) and the number of steps. We do not use partial batches
        step = 2 * self.r + self.m
        num_steps = n // step
        # Find initial values for Q and lam_inv as they are needed to inject low bias exogenous parameters into the
        # model.
        tensors = mv_graph.tensors
        #Q_train = tensors[0]
        Q = tensors[2]
        lam_inv = tensors[3]
        Q_val = session.run(Q)
        lam_inv_val = session.run(lam_inv)
        # Training loop
        indices = np.arange(n)
        feed_dict = {batch_size: self.m} # Batch size never changes in this case
        for e in xrange(self.epochs):
            self.random.shuffle(indices)
            print 'Epoch {}'.format(e)
            if e in self.reduce_lr_epochs:
                lr_val *= self.reduce_lr_factor
            for i in xrange(0, num_steps*step, step):
                # Obtain values of Q and lam_inv for low bias estimation
                feed_dict[mv_graph.low_bias_Q] = Q_val
                feed_dict[mv_graph.low_bias_lam_inv] = lam_inv_val / self.c
                i1 = i + self.m
                i2 = i1 + self.r
                i3 = i2 + self.r
                feed_dict[a] = x[indices[i:i1]]
                feed_dict[a_star1] = x[indices[i1:i2]]
                feed_dict[a_star2] = x[indices[i2:i3]]
                feed_dict[lr] = lr_val
                #session.run(Q_train, feed_dict=feed_dict)
                # Main graph execution (does one step of training updates)
                _, _, Q_val, lam_inv_val, QtQ_val, pa_val, log_fa, A_val, losss_val = session.run(tensors, feed_dict=feed_dict)
            if self.show_variable_during_training:
                self._report(Q_val, lam_inv_val, QtQ_val, feed_dict[a][0] - feed_dict[a_star2][0],
                             pa_val[0], log_fa[0], A_val, losss_val, actual_lamda)

    def _report(self, Q_val, lam_inv_val, QtQ_val, a_val, pa_val, log_fa, A_val, losss_val, actual_lamda):
        actual_scaled = actual_lamda *  (4.0 /(3.0 * self.r)) ** 0.2
        A = np.diag(A_val)
        delta = A - actual_scaled
        #print 'a_diff: {}'.format(a_val)
        #print 'p(a): {}'.format(pa_val)
        #print 'log f(a): {}'.format(log_fa)
        print 'Loss {}\n'.format(losss_val)
        # print 'Eigenvalues: {}\n'.format(1.0 / lam_inv_val)
        # print 'Actual eigenvalues: {}\n'.format(actual_lamda)
        # print 'Q: {}\n'.format(Q_val)
        # print 'A: {}\n'.format(A)
        # print 'Actual {}\n'.format(actual_scaled)
        print 'Delta: {}'.format(delta)
        #print 'mse: {}\n'.format(np.mean(delta ** 2.0))
        # print 'QtQ: {}\n\n'.format(np.diag(QtQ_val))
