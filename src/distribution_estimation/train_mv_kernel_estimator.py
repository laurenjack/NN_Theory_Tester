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
        self.fit_to_underlying_pdf = conf.fit_to_underlying_pdf
        self.epochs = conf.epochs
        self.reduce_lr_epochs = conf.reduce_lr_epochs
        self.reduce_lr_factor = conf.reduce_lr_factor
        self.show_variable_during_training = conf.show_variable_during_training

    def train(self, mv_kde, session, x):
        n = x.shape[0]
        lr_val = self.lr_init
        # Compute the step (total batch size) and the number of steps. We do not use partial batches
        step = 2 * self.r + self.m
        num_steps = n // step
        # Find initial values for Q and lam_inv as they are needed to inject low bias exogenous parameters into the
        # model.
        tensors = mv_kde.tensors
        Q = tensors[0]
        lam_inv = tensors[1]
        Q_val = session.run(Q)
        lam_inv_val = session.run(lam_inv)
        # Training loop
        indices = np.arange(n)
        feed_dict = {mv_kde.batch_size: self.m} # Batch size never changes in this case
        for e in xrange(self.epochs):
            self.random.shuffle(indices)
            print 'Epoch {}'.format(e)
            if e in self.reduce_lr_epochs:
                lr_val *= self.reduce_lr_factor
            for i in xrange(0, num_steps*step, step):
                # Obtain values of Q and lam_inv for low bias estimation
                feed_dict[mv_kde.low_bias_Q] = Q_val
                feed_dict[mv_kde.low_bias_lam_inv] = lam_inv_val / self.c
                i1 = i + self.m
                i2 = i1 + self.r
                i3 = i2 + self.r
                feed_dict[mv_kde.a] = x[indices[i:i1]]
                feed_dict[mv_kde.a_star1] = x[indices[i1:i2]]
                feed_dict[mv_kde.a_star2] = x[indices[i2:i3]]
                feed_dict[mv_kde.lr] = lr_val
                # Main graph execution (does one step of training updates)
                Q_val, lam_inv_val, _, _, QtQ_val, pa_val, fa_val, dist_val = session.run(tensors, feed_dict=feed_dict)
                if self.show_variable_during_training:
                    self._report(Q_val, lam_inv_val, QtQ_val, feed_dict[mv_kde.a][0] - feed_dict[mv_kde.a_star2][0],
                                 pa_val[0], fa_val[0], dist_val[0][0])

    def _report(self, Q_val, lam_inv_val, QtQ_val, a_val, pa_val, fa_val, dist_val):
        print 'a_diff: {}'.format(a_val)
        print 'p(a): {}'.format(pa_val)
        print 'Dist: {}'.format(dist_val)
        print 'f(a): {}'.format(fa_val)
        print 'Eigenvalues: {}'.format(1.0 / lam_inv_val)
        print 'Q: {}\n'.format(Q_val)
        # print 'QtQ: {}\n\n'.format(QtQ_val)
