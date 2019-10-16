import numpy as np


def bp(x, v):
    n = x.shape[0]
    # By the quotient rule df(x) / dsorted_x = 1 / median(x), for non median points
    median = np.median(x)
    dfx_dsorted_x = np.ones(n, dtype=np.float64) / median
    dl_dsorted_x = v * dfx_dsorted_x

    # But, let x_m be the element of x which is the median. A  small change the median won't change f(x_m), which is
    # always f(x_m) = 1. However f'(x_m) != 0, because the point x_m influences f(x_i) for all i != m. That is
    # df(x_i) / d(f(x_m) != 0. We must sum each of these influences on f'(x_m).
    x_sorted = np.sort(x)
    median_index = n // 2
    dl_dsorted_x[median_index] = 0.0
    for i in range(n):
        if i != median_index:
            dl_dsorted_x[median_index] -= v[i] * x_sorted[i] / median ** 2.0

    # Now we have df_dsorted_x, we need to "unsort" to obtain df_dx
    sorted_indices = np.argsort(x)
    dl_dx = np.zeros(n)
    for i in range(n):
        index = sorted_indices[i]
        dl_dx[index] = dl_dsorted_x[i]
    return dl_dx


# x = np.array([3.0, 1.0, 2.0])
# v = (np.arange(3) + 1).astype(np.float64)
# print bp(x, v)

bar = np.arange(1, 199).astype(np.float)
foo = np.arange(1, 100).astype(np.float)
# product = 1
# for i in xrange(50):
#     product *= foo[i]
# print product
log_sum = np.sum(np.log(bar)) - 2*np.sum(np.log(foo))
print log_sum
print np.exp(log_sum)
