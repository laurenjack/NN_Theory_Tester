import tensorflow as tf
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

    def collect(self, kde, fa_tensor, session):
        """
        """
        fa = session.run(fa_tensor, feed_dict={kde.a: self.points_for_graph, kde.a_star1: self.fixed_a_star,
                                           kde.batch_size: self.number_animation_points})
        self.fa_over_time.append(fa)

    def results(self):
        if not self.fa_over_time:
            raise RuntimeError('Must fill f(a) over time, before obtaining the results for animation')
            # Make points for graph 1D again for animation
        return self.points_for_graph[:, 0], self.pa, self.fa_over_time


def create_univariate_collector(conf, random, x, actual_A):
    r = conf.r
    means = conf.means
    sigma_inverse, sigma_determinant = _get_sigma_inverse_and_determinant(actual_A)
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

def _get_sigma_inverse_and_determinant(A):
    sigma = np.matmul(A.transpose(), A)
    sigma_inverse = np.linalg.inv(sigma)
    sigma_determinant = np.linalg.det(sigma)
    return sigma_inverse, sigma_determinant


class NullCollector(object):

    def collect(self, kde, fa_tensor, session):
        pass

    def results(self):
        return None
