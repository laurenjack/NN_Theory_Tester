import numpy as np


def histogram_for_uniform(b, r, n):
    bin_width = 1.0 / float(b)
    edges = [i*bin_width for i in xrange(b+1)]
    x1 = np.random.uniform(size=[n])
    x2 = np.random.uniform(size=[n])
    x3 = np.random.uniform(size=[r])
    h1, _ = np.histogram(x1, bins=edges)
    h2, _ = np.histogram(x2, bins=edges)
    w, _ = np.histogram(x3, bins=edges)
    # h1 = h1 / float(n)
    # h2 = h2 / float(n)
    return h1, h2, np.sum(w * (h1 - h2) ** 2.0) / float(n)


error_sum = 0.0
for i in xrange(100000):
    h1, h2, error = histogram_for_uniform(3, 100, 100)
    error_sum += error
print error_sum / 100000