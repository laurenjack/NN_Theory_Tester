import numpy as np
import tensorflow as tf


def train(kde, conf, session, random, x, collector):
    # Define the positive definite tensor A = RRt
    R_inverse = tf.Variable(np.linalg.inv(conf.R_init), name='R', dtype=conf.float_precision)
    A_inverse_tensor = tf.matmul(R_inverse, tf.transpose(R_inverse))
    loss_tensor = kde.loss(A_inverse_tensor)
    train_op = tf.train.GradientDescentOptimizer(conf.lr).minimize(loss_tensor)
    tf.global_variables_initializer().run()

    # Iteratively train the distribution fitter
    m = conf.m
    r = conf.r
    num_samples = conf.n - 2*r
    batch_count = num_samples // m
    sample_each_epoch = m * batch_count

    indices = np.arange(conf.n)
    for e in xrange(conf.epochs):
        print '\n Epoch: {e}'.format(e=e+1)
        random.shuffle(indices)
        a_star1_indices = indices[0:r]
        a_star2_indices = indices[r:2*r]
        a_star1 = x[a_star1_indices]
        a_star2 = x[a_star2_indices]
        # No partial batches here
        for k in xrange(0, sample_each_epoch, m):
            # Sample a and a_star
            start = 2*r + k
            a_indices = indices[start:start + conf.m]
            a = x[a_indices]

            # Feed to the distribution fitter
            feed_dict = {kde.a: a, kde.a_star1: a_star1, kde.a_star2: a_star2, kde.batch_size: m}
            _, loss, A_inverse = session.run([train_op, loss_tensor, A_inverse_tensor], feed_dict=feed_dict)
            # gradient = gradient[0]
            collector.collect(kde, session)
            if conf.show_A:
                A = np.linalg.inv(A_inverse)
                # determinant_g = np.linalg.det(gradient)
                # gradient_size = np.sum(gradient ** 2.0) ** 0.5
                print 'Loss: {l}\nA: {A}\n'.format(l=loss, A=A)

