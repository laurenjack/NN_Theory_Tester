import numpy as np


def train(kde, conf, session, random, x, collector):
    train_op, cost_op, A_inverse_tensor, gradient_op, fa_op = kde.squared_weighted_mean_error()

    # Iteratively train the distribution fitter
    m = conf.m
    num_samples = conf.n - conf.r
    batch_count = num_samples // m
    sample_each_epoch = m * batch_count

    indices = np.arange(conf.n)
    for e in xrange(conf.epochs):
        print '\n Epoch: {e}'.format(e=e+1)
        random.shuffle(indices)
        a_star_indices = indices[0:conf.r]
        a_star = x[a_star_indices]
        # No partial batches here
        for k in xrange(0, sample_each_epoch, m):
            # Sample a and a_star
            start = conf.r + k
            a_indices = indices[start:start + conf.m]
            a = x[a_indices]

            # Feed to the distribution fitter
            _, cost, A_inverse, gradient, fa = session.run([train_op, cost_op, A_inverse_tensor, gradient_op, fa_op],
                                               feed_dict={kde.a: a, kde.a_star: a_star, kde.batch_size: m})
            gradient = gradient[0]
            collector.collect(kde, session)
            if conf.show_A:
                A = np.linalg.inv(A_inverse)
                determinant_g = np.linalg.det(gradient)
                gradient_size = np.sum(gradient ** 2.0) ** 0.5
                print 'A: {A} \n determinant_g: {d} \n gradient: {g}   fa: {f}'.format(A=A, d=determinant_g,
                                                                                      g=gradient, f=fa)

