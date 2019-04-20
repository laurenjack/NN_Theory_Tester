import numpy as np
import tensorflow as tf


def train(kde, conf, session, random, x, collector):
    # Define the positive definite tensor A = RRt
    R_inverse = tf.Variable(np.linalg.inv(conf.R_init), name='R', dtype=conf.float_precision)
    A_inverse_tensor = tf.matmul(R_inverse, tf.transpose(R_inverse))
    # Placeholder used for the A which determines p(a) estimation
    low_bias_A_inverse = tf.placeholder(dtype=tf.float32, shape=[conf.d, conf.d], name='low_bias_A_inverse')
    if conf.fit_to_underlying_pdf:
        print 'TRAINING ON ACTUAL PDF'
        # loss_tensor, pa_tensor, fa_tensor, distance_tensor = kde.loss(A_inverse_tensor)
        loss_tensor = kde.loss_chi_squared(A_inverse_tensor)
    else:
        print 'Training on data, underlying pdf unknown'
        loss_tensor = kde.loss(A_inverse_tensor, low_bias_A_inverse)

    optimiser = tf.train.GradientDescentOptimizer(conf.lr)
    gradient_var_pairs = optimiser.compute_gradients(loss_tensor)
    new_gradient_var_pairs = []
    for gradient, var in gradient_var_pairs:
        new_gradient = gradient / tf.reduce_mean(tf.abs(gradient))
        new_gradient_var_pairs.append((new_gradient, var))
    train_op = optimiser.apply_gradients(new_gradient_var_pairs)
    tf.global_variables_initializer().run()

    # Iteratively train the distribution fitter
    m = conf.m
    r = conf.r
    c = conf.c
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
            #  Determine A inverse for the low bias estimate
            A_inverse = session.run(A_inverse_tensor)
            # Sample a and a_star
            start = 2*r + k
            a_indices = indices[start:start + conf.m]
            a = x[a_indices]

            # Feed to the distribution fitter
            feed_dict = {kde.a: a, kde.a_star1: a_star1, kde.a_star2: a_star2, low_bias_A_inverse: A_inverse / c,
                         kde.batch_size: m}
            #_, loss, pa, fa, distance = session.run([train_op, loss_tensor, pa_tensor, fa_tensor, distance_tensor], feed_dict=feed_dict)
            _, loss = session.run([train_op, loss_tensor], feed_dict=feed_dict)
            # pa_mean = np.mean(pa)
            # fa_mean = np.mean(fa)
            # gradient = gradient[0]
            collector.collect(kde, session)
            if conf.show_A:
                A= np.linalg.inv(A_inverse) # * np.exp(1)
                # determinant_g = np.linalg.det(gradient)
                # gradient_size = np.sum(gradient ** 2.0) ** 0.5
                print 'Loss: {l}\nA: {A}\n'.format(l=loss, A=A)
                # print 'p(a) mean: {p}   f(a) mean: {f}'.format(p=pa_mean, f=fa_mean)

