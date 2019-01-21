import numpy as np
from data_set import *

WIDTH = 2
NUM_CLASS = 2
BAR_NOISE_PROBABILITY = 0.0
FOO_NOISE_PROBABILITY = 0.0
N_PER_K = 48

if N_PER_K % 4 != 0:
    raise ValueError('Must be divisble by four')
if not (0.0 <= BAR_NOISE_PROBABILITY, FOO_NOISE_PROBABILITY <= 1.0):
    raise ValueError('Must be a valid probabilities')

def training_set():
    num_bar_per_perm = N_PER_K // 2
    num_foo_per_perm = N_PER_K // 4
    bars = _create_bars(num_bar_per_perm)
    foos = _create_foos(num_foo_per_perm)
    bars = _to_data_set(bars, 0, False)
    foos = _to_data_set(foos, 1, False)
    return combine(bars, foos)


def all_value_validation_set():
    """
    Get the all value validation set, this is every possible permutation once with and without noise, as
    well as adversarial examples
    """
    bar = _to_data_set(_all_bar_perms(), 0, False)
    foo = _to_data_set(_all_foo_perms(), 1, False)
    # adv = _to_data_set(_all_unseen_noise_examples(), 1, False)
    return combine(bar, foo) #, adv)


def adversarial_set():
    x = np.array([[[0.75, 0.0], [0.75, 0.0]]])
    y = np.array([1.0])
    is_adversary = np.array([True])
    return _to_data_set(x, y, is_adversary)


def _to_data_set(x, y_scalar, adv_flag):
    n = len(x)
    x = np.concatenate([xi.reshape(1, WIDTH, WIDTH) for xi in x]) # - 0.5
    y = y_scalar * np.ones(n, dtype=np.int32)
    is_adversary = adv_flag * np.ones(n, dtype=np.bool)
    return DataSet(x, y, is_adversary)


def _single_bar(column_to_fill):
    bar = np.zeros((WIDTH, WIDTH), dtype=np.float32)
    bar[:, column_to_fill] = 1.0
    return bar

def _bar_with_possible_noise(column_to_fill):
    bar = _single_bar(column_to_fill)
    if np.random.rand() < FOO_NOISE_PROBABILITY:
        bar[0, 1 - column_to_fill] = 1.0
    return bar


def _single_foo(i, j):
    foo = np.zeros((WIDTH, WIDTH), dtype=np.float32)
    foo[i, j] = 1.0
    return foo


def _all_bar_perms():
    all_perms = []
    for i in xrange(WIDTH):
        bar = _single_bar(i)
        all_perms.append(bar)
        for j in xrange(WIDTH):
            bar_noise = np.copy(bar)
            bar_noise[j, 1 - i] = 1.0
            all_perms.append(bar_noise)
    return all_perms


def _all_foo_perms():
    all_perms = []
    for i in xrange(WIDTH):
        for j in xrange(WIDTH):
            foo = _single_foo(i, j)
            all_perms.append(foo)
            foo_noise = np.copy(foo)
            foo_noise[i, 1 - j] = 1.0
            all_perms.append(foo_noise)
    return all_perms


def _all_unseen_noise_examples():
    I = np.eye(WIDTH)
    I_mirror = 1.0 - I
    return [I, I_mirror]


def _create_bars(num_bar_per_perm):
    left_bar = [_bar_with_possible_noise(0) for i in xrange(num_bar_per_perm)]
    right_bar = [_bar_with_possible_noise(1) for i in xrange(num_bar_per_perm)]
    return left_bar + right_bar


def _create_foos(num_foo_per_perm):
    foos = []
    for i in xrange(num_foo_per_perm):
        for k in xrange(WIDTH):
            for j in xrange(WIDTH):
                foo = _single_foo(k, j)
                # Sprinkle in some noise
                if np.random.rand() < FOO_NOISE_PROBABILITY:
                    foo[k, 1-j] = 1.0
                foos.append(foo)
    return foos


if __name__ == '__main__':
    ts = training_set()
    print ts


