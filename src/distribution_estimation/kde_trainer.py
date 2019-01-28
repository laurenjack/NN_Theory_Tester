import numpy as np


def train(kde, conf, sess, random):
    # Create a Data set
    x = random.normal_numpy_array([conf.n])

    train_op, h_tensor = kde.expected_likelihood()

    # Iteratively train the distribution fitter
    batch_size = conf.m + conf.r
    batch_count = conf.n // batch_size
    sample_each_epoch = batch_size * batch_count

    indices = np.arange(conf.n)
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
            _, h = sess.run([train_op, h_tensor], feed_dict={kde.a: a, kde.a_star: a_star})
            print h
