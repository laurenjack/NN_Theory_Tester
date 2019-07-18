import numpy as np
import tensorflow as tf


class Tranier(object):

    def __init__(self, conf, random):
        self.fit_to_underlying_pdf = conf.fit_to_underlying_pdf
        self.random = random
        self.m = conf.m
        self.r = conf.r
        self.c = conf.c
        self.d = conf.d
        self.epochs = conf.epochs
        self.lr_R = conf.lr_R
        self.reduce_lr_epochs = conf.reduce_lr_epochs
        self.reduce_lr_factor = conf.reduce_lr_factor
        self.R_init = conf.R_init
        self.float_precision = conf.float_precision
        self.show_variable_during_training = conf.show_variable_during_training

    def train_R_for_gaussian_kernel(self, kde, kde_tensors, session, x, collector):
        train_tensor = kde_tensors.train
        loss_tensor = kde_tensors.loss
        fa_tensor = kde_tensors.fa
        A_inverse_tensor = kde_tensors.A_inverse
        A_tensor = kde_tensors.A
        # Iteratively train the distribution fitter
        n = x.shape[0]
        m = self.m
        r = self.r
        lr = self.lr_R
        batches_start = r
        #if not self.fit_to_underlying_pdf:
        batches_start += r
        step = batches_start + m
        batch_count = n // step
        # step = m
        # num_samples = n - batches_start
        # batch_count = num_samples // m # TODO(Jack) deal with uneven batch sizes
        sample_each_epoch = step * batch_count

        indices = np.arange(n)
        for e in xrange(self.epochs):
            self.random.shuffle(indices)
            if e in self.reduce_lr_epochs:
                lr *= self.reduce_lr_factor
            feed_dict = {kde.batch_size: m, kde.lr: lr}
            # a_star1_indices = indices[0:r]
            #feed_dict = {kde.a_star1: x[a_star1_indices], kde.lr: lr}
            if not self.fit_to_underlying_pdf:
                # a_star2_indices = indices[r:2*r]
                # feed_dict[kde.a_star2] = x[a_star2_indices]
                # Grab the value of A_inverse, for a low biased substitution. Devide by c because we are using the
                # inverse
                A_Inverse = session.run(A_inverse_tensor)
                feed_dict[kde.low_bias_A_inverse] = A_Inverse / self.c
                # Update learning rate stepwise
            # No partial batches here
            for k in xrange(0, sample_each_epoch, step):
                # # Sample a
                # start = batches_start + k
                # a_indices = indices[start:start + m]
                # a = x[a_indices]
                # feed_dict[kde.a] = a
                # feed_dict[kde.batch_size] = m
                feed_dict[kde.a] = x[indices[k:k + m]]
                feed_dict[kde.a_star1] = x[indices[k+m:k+m+r]]
                feed_dict[kde.a_star2] = x[indices[k+m+r:k+m+2*r]]
                _, loss, A, fa = session.run([train_tensor, loss_tensor, A_tensor, fa_tensor], feed_dict=feed_dict)
                collector.collect(A_tensor, session)
                if self.show_variable_during_training:
                    print '\n Epoch: {e}'.format(e=e + 1)
                    print 'Loss: {l}\n{A}'.format(l=loss, A=A[0,0:5])
                    mean_var = 0.0
                    for i in xrange(self.d):
                        mean_var += A[i,i]
                    mean_var /= float(self.d)
                    print mean_var
                    print 'Mean pa: {}\n'.format(np.mean(fa))
        return A
