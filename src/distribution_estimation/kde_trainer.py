import numpy as np


def train(kde, conf, sess, random):
    # Create a Data set
    x = random.normal_numpy_array([conf.n])

    train_op, cost_op, h_tensor, gradient_op = kde.squared_weighted_mean_error()

    # Iteratively train the distribution fitter
    batch_size = conf.m + conf.r
    batch_count = conf.n // batch_size
    sample_each_epoch = batch_size * batch_count

    indices = np.arange(conf.n)
    cost_average = 0.0
    for e in xrange(conf.epochs):
        random.shuffle(indices)
        # No partial batches here
        for k in xrange(0, sample_each_epoch, batch_size):
            # Sample a and a_star
            star_start = k + conf.m
            a_indices = indices[k:star_start]
            a = x[a_indices]
            a_star_indices = indices[star_start:star_start + conf.r]
            a_star = x[a_star_indices]

            # Feed to the distribution fitter
            _, cost, h, gradient = sess.run([train_op, cost_op, h_tensor, gradient_op], feed_dict={kde.a: a, kde.a_star: a_star})
            print 'h: {h}   cost: {c}   h_gradient: {g}'.format(h=h, c=cost, g=gradient)
            cost_average += cost
    cost_average = cost_average / (conf.epochs * batch_count)
    print '\nCost Average: {c}'.format(c=cost_average)
