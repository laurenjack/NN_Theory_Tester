import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import patches


class UnivariateAnimator(object):

    def __init__(self, conf):
        self.animation_interval = conf.animation_interval

    def animate_density(self, a, pa, fa_over_time):
        figure, _ = plt.subplots()
        # Plot the actual underlying distribution p(a), which is fixed.
        plt.plot(a, pa)

        # Define the animation callbacks
        kde_points = plt.scatter(a, fa_over_time[0])

        def init():
            return kde_points

        def update(frame):
            next_fa = fa_over_time[frame + 1]
            points = np.array([a, next_fa])
            number_animation_points = points.shape[1]
            kde_points.set_offsets(np.split(points, number_animation_points, axis=1))

        number_of_steps = len(fa_over_time) - 1
        ani = animation.FuncAnimation(figure, update, frames=number_of_steps, init_func=init,
                                      interval=self.animation_interval, repeat=False)
        plt.show()


class TwoDAnimator(object):

    def __init__(self, conf, actual_A):
        self.axis_max_and_min = conf.axis_max_and_min
        self.animation_interval = conf.animation_interval
        self.means = conf.means
        self.concentric_gap = conf.concentric_gap
        self.max_deviations = conf.max_deviations
        # Get the eigen-values and vectors of the standard deviation matrix, as the ellipse function requires
        eigenvalues, Q = np.linalg.eig(actual_A)
        self.base_width = 2.0 * eigenvalues[0]
        self.base_height = 2.0 * eigenvalues[1]
        first_axis = Q[:, 0]
        i = np.array([1.0, 0.0])
        cos_theta = first_axis.dot(i)
        self.angle = math.acos(cos_theta) / math.pi * 180
        # Account for where x and y are negatively correlated, in this case the angle direction is flipped
        if first_axis[1] < 0:
            self.angle = 360 - self.angle

    def animate_density(self, fixed_a_star, z, A_over_time):
        figure, axis = plt.subplots()
        axis.set_xlim(-self.axis_max_and_min, self.axis_max_and_min)
        axis.set_ylim(-self.axis_max_and_min, self.axis_max_and_min)
        # Plot the concentric ellipses showing the actual underlying Gaussians
        number_of_ellipses = int(math.floor(self.max_deviations / self.concentric_gap))
        number_of_means = self.means.shape[0]
        for scale in xrange(number_of_ellipses):
            width = self.base_width * scale * self.concentric_gap
            height = self.base_height * scale * self.concentric_gap
            for i in xrange(number_of_means):
                mean = self.means[i]
                ellipse = patches.Ellipse(xy=mean, width=width, height=height, fill=False, angle=self.angle, color='k')
                axis.add_artist(ellipse)
                ellipse.set_visible(True)

        # Setup animation of rbf over time
        scaled = self._scale_random_kde_draws(fixed_a_star, z, A_over_time[0])
        x, y = scaled[:, 0], scaled[:, 1]
        scaled_kde_scatter = plt.scatter(x, y)

        def init():
            return scaled_kde_scatter

        def update(frame):
            A = A_over_time[frame + 1]
            scaled = self._scale_random_kde_draws(fixed_a_star, z, A)
            r = scaled.shape[0]
            scaled_kde_scatter.set_offsets(np.split(scaled, r, axis=0))

        number_of_steps = len(A_over_time) - 1
        ani = animation.FuncAnimation(figure, update, frames=number_of_steps, init_func=init,
                                      interval=self.animation_interval, repeat=False)
        plt.show()


    def _scale_random_kde_draws(self, fixed_a_star, z, A):
        return fixed_a_star + np.matmul(z, A)


