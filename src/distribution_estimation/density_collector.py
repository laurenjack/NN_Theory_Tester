import numpy as np
import math


_SIGMA = 5.0


class UnivariateCollector(object):

    def __init__(self, fixed_a, fixed_a_star, pa):
        self.number_animation_points = fixed_a.shape[0]
        self.fixed_a = fixed_a
        self.fixed_a_star = fixed_a_star
        self.pa = pa
        self.fa_over_time = []

    def collect(self, kde, session):
        """
        """
        fa_op = kde.pdf()
        fa = session.run(fa_op, feed_dict={kde.a: self.fixed_a, kde.a_star: self.fixed_a_star,
                                           kde.batch_size: self.number_animation_points})
        self.fa_over_time.append(fa)

    def results(self):
        """
        """
        if not self.fa_over_time:
            raise RuntimeError('Must fill f(a) over time, before obtaining the results for animation')
        return self.fixed_a, self.pa, self.fa_over_time

    def _kde(self, reference_set, h, x):
        m = x.shape[0]
        r = reference_set.shape[0]
        x.reshape(m, 1) - reference_set.reshape(1, r)


def create_univariate_collector(conf, random, x):
    r = conf.r
    means = conf.means
    standard_deviation = conf.standard_deviation
    number_of_animation_points = conf.number_of_animation_points
    # Find the boundaries on the animation points
    upper_bound = max(means) + _SIGMA
    lower_bound = min(means) - _SIGMA
    width = upper_bound - lower_bound
    # Generate equally spaced animation points
    fixed_a = np.arange(0, number_of_animation_points) / float(number_of_animation_points)
    fixed_a = lower_bound + fixed_a * float(width)
    # Select a fixed set of reference points to stick with
    fixed_a_star = random.choice(x, r)

    pa = _compute_actual(fixed_a, means, standard_deviation)

    return UnivariateCollector(fixed_a, fixed_a_star, pa)


def _compute_actual(fixed_a, means, standard_deviation):
    """ Compute the relative likelihoods from the actual distribution.
    """
    number_of_animation_points = fixed_a.shape[0]
    pa = np.zeros(number_of_animation_points, dtype=np.float32)
    for mean in means:
        exponent = -(fixed_a - mean) ** 2.0 / (2.0 * standard_deviation ** 2.0)
        local_p = np.exp(exponent)
        pa += local_p
    number_of_means = float(len(means))
    return 1.0 / ((2.0 * math.pi) ** 0.5 * standard_deviation * number_of_means) * pa