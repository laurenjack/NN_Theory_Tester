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
    means = conf.means
    sigma_inverse, sigma_determinant = _get_sigma_inverse_and_determinant(conf)
    number_of_animation_points = conf.number_of_animation_points
    # Find the boundaries on the animation points
    upper_bound = means[:, 0].max() + _SIGMA
    lower_bound = means[:, 0].min() - _SIGMA
    width = upper_bound - lower_bound
    # Generate equally spaced animation points
    points_for_graph = np.arange(0, number_of_animation_points) / float(number_of_animation_points)
    points_for_graph = lower_bound + points_for_graph * float(width)
    points_for_graph = points_for_graph.reshape([number_of_animation_points, 1])
    # Select a fixed set of reference points to stick with
    fixed_a_star = random.choice(x, r)

    pa = _compute_actual(points_for_graph, means, sigma_inverse, sigma_determinant)

    return UnivariateCollector(points_for_graph, fixed_a_star, pa)


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


class MeanSquaredErrorCollector(object):
    """Reports on the mean squared error of the actual distribution less that given by the kde, i.e. p(x) - f(x).

    This collector is only applicable where the actual function p(x) is a known mixture of gaussians
    """

    def __init__(self, conf, random, x):
        self.m = conf.m
        self.random = random
        self.means = conf.means
        self.sigma_inverse, self.sigma_determinant = _get_sigma_inverse_and_determinant(conf)
        
        # Copy as shuffle has a side effect
        x_copy = np.copy(x)
        random.shuffle(x_copy)
        r = conf.r
        self.fixed_a_star = x_copy[0:r]
        self.a_all = x[r:]

    def collect(self, kde, session):
        # Get a batch of samples to compute the cost for
        a = self.random.choice(self.a_all, self.m)

        # Get kde pdf likelihoods
        pdf = kde.pdf()
        fa = session.run(pdf, feed_dict={kde.a: a, kde.a_star: self.fixed_a_star, kde.batch_size: self.m})
        # Get the real likelihoods
        pa = _compute_actual(a, self.means, self.sigma_inverse, self.sigma_determinant)

        mean_squared_error = np.mean((pa - fa) ** 2.0)
        print 'Mean Squared Error Against Actual: {mse}\n'.format(mse=mean_squared_error)


class NullCollector(object):

    def collect(self, kde, session):
        pass

    def results(self):
        return None


def _compute_actual(points_for_graph, means, sigma_inverse, sigma_determinant):
    """ Compute the relative likelihoods from the actual distribution.
    """
    number_of_means, d = means.shape
    n = points_for_graph.shape[0]

    points_for_graph = points_for_graph.reshape(n, 1, d)
    means = means.reshape(1, number_of_means, d)
    difference = points_for_graph - means
    distance = np.matmul(difference, sigma_inverse).reshape(n, number_of_means, 1, d)
    distance = np.matmul(distance, difference.reshape(n, number_of_means, d, 1))
    distance = distance.reshape(n, number_of_means)
    pa_unnormed = np.sum(np.exp(-0.5 * distance), axis=1)
    pa = 1.0 / (((2.0 * math.pi) ** d * sigma_determinant) ** 0.5 * number_of_means) * pa_unnormed
    return pa


def _get_sigma_inverse_and_determinant(conf):
    A = conf.actual_A
    sigma = np.matmul(A.transpose(), A)
    sigma_inverse = np.linalg.inv(sigma)
    sigma_determinant = np.linalg.det(sigma)
    return sigma_inverse, sigma_determinant
