import numpy as np
import tensorflow as tf


class Tranier(object):

    def __init__(self, conf, random):
        self.random = random
        self.n = conf.n
        self.m = conf.m
        self.r = conf.r
        self.c = conf.c
        self.d = conf.d
        self.epochs = conf.epochs
        self.lr_A = conf.lr_A
        self.lr_h = conf.lr_h
        self.A_init = conf.A_init
        self.h_init = conf.h_init
        self.float_precision = conf.float_precision
        self.show_variable_during_training = conf.show_variable_during_training

    def train_H(self, kde, session, x, collector):
        # Define the positive definite tensor sigma = AAt
        A_inverse_tensor = tf.Variable(np.linalg.inv(self.A_init), name='A', dtype=self.float_precision)
        H_inverse_tensor = tf.matmul(A_inverse_tensor, tf.transpose(A_inverse_tensor))
        loss_tensor = kde.loss_for_estimating_H(H_inverse_tensor)
        self._train(kde, loss_tensor, tf.matrix_inverse(H_inverse_tensor), session, x, collector, self.lr_A)
        return session.run(H_inverse_tensor)

    def train_h(self, kde, session, x, collector, trained_H_inverse):
        h_tensor = tf.Variable(self.h_init, name='h', dtype=self.float_precision)
        loss_tensor = kde.loss_for_chi_squared_bandwidth(trained_H_inverse, h_tensor)
        self._train(kde, loss_tensor, h_tensor, session, x, collector,  self.lr_h)
        return session.run(h_tensor)

    def _train(self, kde, loss_tensor, tensor_to_watch, session, x, collector, lr):
        optimiser = tf.train.GradientDescentOptimizer(lr)
        gradient_var_pairs = optimiser.compute_gradients(loss_tensor)
        new_gradient_var_pairs = []
        for gradient, var in gradient_var_pairs:
            new_gradient = gradient / tf.reduce_mean(tf.abs(gradient))
            new_gradient_var_pairs.append((new_gradient, var))
        train_op = optimiser.apply_gradients(new_gradient_var_pairs)
        tf.global_variables_initializer().run()

        # Iteratively train the distribution fitter
        n = self.n
        m = self.m
        r = self.r
        c = self.c
        num_samples = n - 2 * r
        batch_count = num_samples // m
        sample_each_epoch = m * batch_count

        indices = np.arange(n)
        for e in xrange(self.epochs):
            print '\n Epoch: {e}'.format(e=e + 1)
            self.random.shuffle(indices)
            a_star1_indices = indices[0:r]
            a_star2_indices = indices[r:2 * r]
            a_star1 = x[a_star1_indices]
            a_star2 = x[a_star2_indices]
            # No partial batches here
            for k in xrange(0, sample_each_epoch, m):
                # Sample a and a_star
                start = 2 * r + k
                a_indices = indices[start:start + m]
                a = x[a_indices]

                # Feed to the distribution fitter
                feed_dict = {kde.a: a, kde.a_star1: a_star1, kde.a_star2: a_star2, kde.batch_size: m}
                _, loss, to_watch = session.run([train_op, loss_tensor, tensor_to_watch], feed_dict=feed_dict)
                collector.collect(tensor_to_watch, session)
                if self.show_variable_during_training:
                    print 'Loss: {l}\n{to_watch}\n'.format(l=loss, to_watch=to_watch[0,0])

        #
        # if conf.fit_to_underlying_pdf:
        #     print 'TRAINING ON ACTUAL PDF'
        #     # loss_tensor, pa_tensor, fa_tensor, distance_tensor = kde.loss(A_inverse_tensor)
        #     loss_tensor = kde.loss_for_chi_squared_kernels(H_inverse_tensor)
        # else:
        #     print 'Training on data, underlying pdf unknown'
