import tensorflow as tf
import numpy as np
import math

class ChiSquaredCollector():

    def __init__(self, H_inverse, evenly_spaced_distances, points_for_graph, fixed_a_star, pa, pdf_functions):
        self.number_animation_points = points_for_graph.shape[0]
        self.evenly_spaced_distances = evenly_spaced_distances
        self.points_for_graph = points_for_graph
        self.fixed_a_star = fixed_a_star
        self.pa = pa
        self.fa_over_time = []
        self.pdf_functions = pdf_functions
        self.H_inverse = H_inverse

    def collect(self, h, session):
        """
        """
        fa_op = self.pdf_functions.chi_square_kde_centered_exponent(self.H_inverse , self.points_for_graph,
                                                                    self.fixed_a_star, self.number_animation_points, h)
        fa = session.run(fa_op)
        self.fa_over_time.append(fa)

    def results(self):
        """
        """
        if not self.fa_over_time:
            raise RuntimeError('Must fill f(a) over time, before obtaining the results for animation')
        # Make points for graph 1D again for animation
        return self.evenly_spaced_distances, self.pa, self.fa_over_time


def create_chi_squared_collector(conf, random, H_inverse, x, data_generator, pdf_functions, session):
    r = conf.r
    d = conf.d
    # Convert means to 1D
    means = conf.means
    number_of_animation_points = conf.number_of_animation_points
    # Find the boundaries on the animation points
    upper_bound = 2 * d
    lower_bound = 0
    evenly_spaced_distances = _evenly_spaced_points(lower_bound, upper_bound, number_of_animation_points)
    # Draw random points and scale to fit distances
    animation_points, _ = data_generator.sample_gaussian_mixture(number_of_animation_points)
    sampled_distance = np.sum(animation_points ** 2.0, axis=1) ** 0.5
    animation_points = animation_points / sampled_distance.reshape(number_of_animation_points, 1)
    animation_points = animation_points * evenly_spaced_distances.reshape(number_of_animation_points, 1) ** 0.5
    # Select a fixed set of reference points to stick with
    fixed_a_star = random.choice(x, r)

    chi_squared_tensor = pdf_functions.chi_squared_distribution(evenly_spaced_distances)
    pa = session.run(chi_squared_tensor)

    return ChiSquaredCollector(H_inverse, evenly_spaced_distances, animation_points, fixed_a_star, pa, pdf_functions)


class NullCollector(object):

    def collect(self, kde, session):
        pass

    def results(self):
        return None


def _evenly_spaced_points(lower_bound, upper_bound, number_of_animation_points):
    width = upper_bound - lower_bound
    # Generate equally spaced animation points
    evenly_spaced_points = np.arange(0, number_of_animation_points) / float(number_of_animation_points)
    evenly_spaced_points = lower_bound + evenly_spaced_points * float(width)
    return evenly_spaced_points.astype(np.float32)
