import numpy as np
import math


_SIGMA = 5.0


class UnivariateCollector(object):

    def __init__(self, points_for_graph, fixed_a_star, pa):
        self.number_animation_points = points_for_graph.shape[0]
        self.points_for_graph = points_for_graph
        self.fixed_a_star = fixed_a_star
        self.pa = pa
        self.fa_over_time = []

    def collect(self, kde, session):
        """
        """
        fa_op = kde.pdf()
        fa = session.run(fa_op, feed_dict={kde.a: self.points_for_graph, kde.a_star: self.fixed_a_star,
                                           kde.batch_size: self.number_animation_points})
        self.fa_over_time.append(fa)

    def results(self):
        """
        """
        if not self.fa_over_time:
            raise RuntimeError('Must fill f(a) over time, before obtaining the results for animation')
        # Make points for graph 1D again for animation
        return self.points_for_graph[:, 0], self.pa, self.fa_over_time

    def _kde(self, reference_set, h, x):
        m = x.shape[0]
        r = reference_set.shape[0]
        x.reshape(m, 1) - reference_set.reshape(1, r)


def create_univariate_collector(conf, random, x):
    r = conf.r
    # Convert means to 1D
    means = conf.means[:, 0]
    # In the univariate case, we use 1D arrays to graph the actual underlying distribution
    standard_deviation = conf.actual_A[0, 0]
    number_of_animation_points = conf.number_of_animation_points
    # Find the boundaries on the animation points
    upper_bound = means.max() + _SIGMA
    lower_bound = means.min() - _SIGMA
    width = upper_bound - lower_bound
    # Generate equally spaced animation points
    points_for_graph = np.arange(0, number_of_animation_points) / float(number_of_animation_points)
    points_for_graph = lower_bound + points_for_graph * float(width)
    # Select a fixed set of reference points to stick with
    fixed_a_star = random.choice(x, r)

    pa = _compute_actual(points_for_graph, means, standard_deviation)

    points_for_graph_reshaped = points_for_graph.reshape([number_of_animation_points, 1])
    return UnivariateCollector(points_for_graph_reshaped, fixed_a_star, pa)


class MultivariateCollector(object):

    def __init__(self, fixed_a_star, z):
        self.fixed_a_star = fixed_a_star
        self.z = z
        self.A_over_time = []

    def collect(self, kde, session):
        """
        """
        A_tensor = kde.A
        A = session.run(A_tensor)
        # Observe the scaling for the new A (technically this isn't a random draw from the rbf, we're drawing from
        # each reference point, but doing so will allow us to focus on changes in A).
        animation_points = self.fixed_a_star + np.matmul(self.z, A)
        self.A_over_time.append(A)

    def results(self):
        return self.fixed_a_star, self.z, self.A_over_time


def create_multivariate_collector(conf, random, x):
    r = conf.r
    d = conf.d
    # Draw a fixed set of reference points
    fixed_a_star = random.choice(x, r, replace=False)
    # Make a standard normal draw for each reference point, this will stay remain over the couyrse of the animation,
    # the only change will be the scale due to A
    z = random.normal_numpy_array([r, d])
    return MultivariateCollector(fixed_a_star, z)


class NullCollector(object):

    def collect(self, kde, session):
        pass

    def results(self):
        return None


def _compute_actual(points_for_graph, means, standard_deviation):
    """ Compute the relative likelihoods from the actual distribution.
    """
    number_of_animation_points = points_for_graph.shape[0]
    pa = np.zeros(number_of_animation_points, dtype=np.float32)
    number_of_means = means.shape[0]
    for i in xrange(number_of_means):
        mean = means[i]
        exponent = -(points_for_graph - mean) ** 2.0 / (2.0 * standard_deviation ** 2.0)
        local_p = np.exp(exponent)
        pa += local_p
    return 1.0 / ((2.0 * math.pi) ** 0.5 * standard_deviation * number_of_means) * pa
