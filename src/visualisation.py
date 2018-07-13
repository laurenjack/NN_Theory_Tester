import matplotlib.pyplot as plt
import numpy as np

def plot(title, x, y):
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.title(title)
    plt.plot(x, y)
    plt.show()

def plot_all(X, predicted, actual):
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

def _plot_image(im_vector, predicted, actual, i):
    fig = plt.figure(i+1)
    ax = fig.gca()
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    plt.title('Predicted: ' + str(predicted), loc='left')
    plt.title('Actual: ' + str(actual), loc='right')
    plt.imshow(im_vector.reshape(28, 28), cmap='gray', interpolation='nearest')

def _plot_sub_image(im_vector, sub_num):
    sub_plot = plt.subplot(1, 2, sub_num)
    sub_plot.yaxis.set_visible(False)
    sub_plot.xaxis.set_visible(False)
    sub_plot.imshow(im_vector.reshape(28, 28), interpolation='nearest')