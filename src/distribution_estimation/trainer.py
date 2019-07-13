import numpy as np
import tensorflow as tf


class Tranier(object):

    def __init__(self, conf, random):
        self.random = random
        self.m = conf.m
        self.r = conf.r
        self.c = conf.c
        self.d = conf.d
        self.epochs = conf.epochs
        self.lr_R = conf.lr_R
        self.R_init = conf.R_init
        self.float_precision = conf.float_precision
        self.show_variable_during_training = conf.show_variable_during_training

    def train_R_for_gaussian_kernel(self, kde, session, x, collector):
        # Define the positive definite tensor A = RRt
        R_inverse_tensor = tf.Variable(np.linalg.inv(self.R_init), name='R_inverse', dtype=self.float_precision)
        A_inverse_tensor = tf.matmul(R_inverse_tensor, tf.transpose(R_inverse_tensor))
        loss_tensor, _, fa_tensor = kde.loss(A_inverse_tensor)
        self._train(kde, loss_tensor, tf.matrix_inverse(A_inverse_tensor), session, x, collector, self.lr_R, fa_tensor)

    def _train(self, kde, loss_tensor, tensor_to_watch, session, x, collector, lr, fa_tensor):
        optimiser = tf.train.GradientDescentOptimizer(lr)
        gradient_var_pairs = optimiser.compute_gradients(loss_tensor)
        new_gradient_var_pairs = []
        for gradient, var in gradient_var_pairs:
            new_gradient = gradient / tf.reduce_mean(tf.abs(gradient))
            new_gradient_var_pairs.append((new_gradient, var))
        train_op = optimiser.apply_gradients(new_gradient_var_pairs)
        tf.global_variables_initializer().run()

        # Iteratively train the distribution fitter
        n = x.shape[0]
        m = self.m
        r = self.r
        c = self.c
        num_samples = n - r
        batch_count = num_samples // m
        sample_each_epoch = m * batch_count

        indices = np.arange(n)
        for e in xrange(self.epochs):
            print '\n Epoch: {e}'.format(e=e + 1)
            self.random.shuffle(indices)
            a_star1_indices = indices[0:r]
            a_star1 = x[a_star1_indices]
            # No partial batches here
            for k in xrange(0, sample_each_epoch, m):
                # Sample a and a_star
                start = r + k
                a_indices = indices[start:start + m]
                a = x[a_indices]

                # Feed to the distribution fitter
                feed_dict = {kde.a: a, kde.a_star1: a_star1, kde.batch_size: m}
                _, loss, to_watch, fa = session.run([train_op, loss_tensor, tensor_to_watch, fa_tensor], feed_dict=feed_dict)
                collector.collect(tensor_to_watch, session)
                if self.show_variable_during_training:
                    print 'Loss: {l}\n{to_watch}'.format(l=loss, to_watch=to_watch[0,0:5])
                    mean_var = 0.0
                    for i in xrange(self.d):
                        mean_var += to_watch[i,i]
                    mean_var /= float(self.d)
                    print mean_var
                    print 'Mean pa: {}\n'.format(np.mean(fa))
