import numpy as np


def train(kde, conf, session, random, x, collector):
    train_op, cost_op, h_tensor, gradient_op = kde.squared_weighted_mean_error()

    # Iteratively train the distribution fitter
    m = conf.m
    num_samples = conf.n - conf.r
    batch_count = num_samples // m
    sample_each_epoch = m * batch_count

    indices = np.arange(conf.n)
    costs = []
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
            _, cost, h, gradient = session.run([train_op, cost_op, h_tensor, gradient_op],
                                               feed_dict={kde.a: a, kde.a_star: a_star, kde.batch_size: m})
            collector.collect(kde, session)
            print 'h: {h}   cost: {c}   h_gradient: {g}'.format(h=h, c=cost, g=gradient)
            costs.append(cost)
    cost_average = sum(costs) / (conf.epochs * batch_count)
    print '\nCost Average: {c}'.format(c=cost_average)
    cost_max = max(costs)
    print 'Max cost: {m}'.format(m=cost_max)
