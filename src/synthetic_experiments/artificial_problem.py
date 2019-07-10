import numpy as np

from src.data_set import Dataset
from src.rbf_softmax.configuration import conf

DATA_START = -3.0
DATA_END = 3.0
DATA_WIDTH = DATA_END - DATA_START

# TODO(Jack) does not properly use data_set contract
def simple_identical_plane(n_per_class, d, K): # , keep_individual_dims_constant=False
    """Generate K classes each seperated by an identical plane with biases"""
    if conf.n != n_per_class * K:
        raise ValueError('n must factorise to n_per_class * K')
    dir = np.random.randn(d)
    dir /= np.sum(dir ** 2.0) ** 0.5

    step = DATA_WIDTH / K
    biases = np.arange(DATA_START, DATA_END, step)

    starts = [0.3 * step] + [DATA_START] * (d - 1)
    ends = [0.7 * step] + [DATA_END] * (d - 1)
    uniform_variation = np.random.uniform(starts, ends, size=(n_per_class*K, d)) # , DATA_START , DATA_END

    #Apply rotation to uniform variation, so that each of these vectors points in the norm from the plane
    basis_for_plane = np.eye(d)[:, :-1]
    forced = basis_for_plane[:-1].dot(-dir[:-1]) / dir[-1]
    basis_for_plane[-1] = forced
    basis_for_plane /= np.sum(basis_for_plane ** 2.0, axis=0) ** 0.5
    rotation = np.concatenate([dir.reshape(d, 1), basis_for_plane], axis=1)
    rotated_uniform_variation = uniform_variation.dot(rotation.transpose())

    rotated_uniform_variation = rotated_uniform_variation.reshape((n_per_class, K, d))
    rotated_uniform_variation = rotated_uniform_variation.transpose((0, 2, 1))
    X = dir.reshape(1, d, 1) * biases.reshape(1, 1, K) + rotated_uniform_variation
    X = X.transpose(0, 2, 1)
    X = X.reshape(n_per_class * K, d)

    y = np.arange(K)
    y = np.tile(y, [n_per_class, 1])
    y = y.transpose()
    y = y.flatten()
    return Dataset(X, y, X, y)

if __name__ == '__main__':
    n_per_class = 50
    d = 50
    K = 4
    conf.n = n_per_class * K
    data_set = simple_identical_plane(n_per_class, d, K)
    X = data_set.train.x
    Xt = X.reshape(n_per_class, K, d).transpose(1, 2, 0)
    d0 = Xt[:, 0]
    d1 = Xt[:, 1]
    import src.visualisation
    src.visualisation.scatter_all('All x', d0, d1)
