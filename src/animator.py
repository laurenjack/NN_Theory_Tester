import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
from matplotlib.patches import Ellipse

def animate(zs, z_bars, taus, conf):
    show_details = conf.show_details
    show_animation = conf.show_animation
    fig, ax = plt.subplots()
    #_draw(clusters)
    epochs = len(zs[0])
    ax.set_xlim(-50.0, 50.0)
    ax.set_ylim(-50.0, 50.0)

    #Reshape the z bars for drawing
    init_z_bar = z_bars[0]
    d = init_z_bar.shape[0]
    num_classes = init_z_bar.shape[1]
    z_bars, taus = _prep_z_bars(z_bars, taus, num_classes, d)

    # Build a colour function based on the number of classes
    color_func = _get_cmap(num_classes)

    # Plot the inital centres
    scats = []
    to_draw = []
    #fig = plt.figure(1)
    for k in xrange(num_classes):
        z = zs[k]
        z_bar = z_bars[k]
        tau = taus[k]
        z_init = z[0]
        z_bar_init = z_bar[0]
        tau_init = tau[0]
        colour = color_func(k)
        scat_z = _plot_scatter(z_init, colour, fig)
        scat_z_bar = _plot_scatter(z_bar_init, colour, fig, tau=tau_init, marker='x')
        scats.append(scat_z)
        scats.append(scat_z_bar)
        to_draw.append(z)
        to_draw.append(z_bar)

    #Delegates required for the animator
    def init():
        return scats

    def update(frame):
        for scat, all_prev in zip(scats, to_draw):
            current_points = all_prev[frame+1]
            num_points, _ = current_points.shape
            as_list = np.split(current_points, num_points, axis=0)
            scat.set_offsets(as_list)

    #Run the animation
    if show_animation:
        ani = FuncAnimation(fig, update, frames=epochs - 1, init_func=init, interval=500, repeat=False)
    #plt.show()
    if show_details:
        _show_details_1_culster(zs[0], z_bars[0], taus[0])
    plt.show()

def _prep_z_bars(z_bars, taus, num_class, d):
    z_bar_new = []
    tau_new = []
    for k in xrange(num_class):
        z_bar_new.append([])
        tau_new.append([])
        for z_bar, tau in zip(z_bars, taus):
            z_bar_reshaped = z_bar[:, k].reshape(1, d)
            tau_reshaped = tau[:, k].reshape(1, d)
            z_bar_new[k].append(z_bar_reshaped)
            tau_new[k].append(tau_reshaped)
    return z_bar_new, tau_new



# def _draw(clusters):
#     K = len(clusters)
#     #Get even spread of colours for scatter plot
#     color_func = _get_cmap(K)
#     for i in xrange(K):
#         cluster = clusters[i]
#         _plot_scatter(cluster, color_func(i))

def _show_details_1_culster(zs, z_bar_list, tau_list):
    epochs = len(zs)
    t = np.arange(epochs)
    plt.figure(2)
    m = zs[0].shape[0]
    for i in xrange(m):
        z_means = [z[i][0] for z in zs]
        plt.plot(t, z_means)
    plt.figure(3)
    for i in xrange(m):
        z_means = [z[i][1] for z in zs]
        plt.plot(t, z_means)
    plt.figure(4)
    z_bar_means = [z_bar[0] for z_bar in z_bar_list]
    plt.plot(t, z_bar_means)
    plt.figure(5)
    tau_means = [tau[0] for tau in tau_list]
    plt.plot(t, tau_means)

def _plot_scatter(clust_or_centres, colour, fig, tau=None, marker=None):
    xs = clust_or_centres[:, 0]
    ys = clust_or_centres[:, 1]
    if marker is None:
        return plt.scatter(xs, ys, color=colour)
    centre = clust_or_centres[0]
    taus = tau[0]
    ell = Ellipse(xy=centre, width=1.0 /taus[0], height=1.0/taus[1], fill=False)
    ax = fig.gca()
    ax.add_artist(ell)
    return plt.scatter(xs, ys, color=colour, marker=marker)


def _get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='gist_rainbow')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color