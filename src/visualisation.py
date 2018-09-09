import matplotlib.pyplot as plt
import numpy as np
import scipy.misc


def plot(title, x, y):
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.title(title)
    plt.plot(x, y)
    plt.show()

def scatter_all(title, x, y):
    ax = plt.gca()
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    plt.title(title)
    for i in xrange(x.shape[0]):
        plt.scatter(x[i], y[i])
    plt.show()

def plot_all_image(X, predicted, actual):
    n = X.shape[0]
    for i in xrange(n):
        _plot_image(X[i], predicted[i], actual[i], i)
    plt.show()

def plot_all_with_originals(X, predicted, actual, originals):
    for i in xrange(originals.shape[0]):
        plt.figure(i + 1)
        org = originals[i]
        adv = X[i]
        _plot_sub_image(org, 1)
        _plot_sub_image(adv, 2)
    plt.show()

def plot_histogram(x, bins=100):
    plt.hist(x, bins=bins)
    plt.show()

def _plot_image(im_vector, predicted, actual, i):
    fig = plt.figure(i+1)
    ax = fig.gca()
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    plt.title('Predicted: ' + str(predicted), loc='left')
    plt.title('Actual: ' + str(actual), loc='right')
    shape = im_vector.shape
    if len(shape) > 2:
        scipy.misc.imshow(im_vector)
    else:
        plt.imshow(im_vector.reshape(28, 28), cmap='gray', interpolation='nearest')

def _plot_sub_image(im_vector, sub_num):
    sub_plot = plt.subplot(1, 2, sub_num)
    sub_plot.yaxis.set_visible(False)
    sub_plot.xaxis.set_visible(False)
    sub_plot.imshow(im_vector.reshape(28, 28), interpolation='nearest')