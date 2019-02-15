import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Animator(object):

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
            m = points.shape[1]
            kde_points.set_offsets(np.split(points, m, axis=1))

        number_of_steps = len(fa_over_time) - 1
        ani = animation.FuncAnimation(figure, update, frames=number_of_steps, init_func=init,
                                      interval=self.animation_interval, repeat=False)
        plt.show()


