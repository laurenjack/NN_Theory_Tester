import numpy as np


def generate_gaussian_mixture(conf, random):
    """Generates an [n, d], where the n elements are drawn from a mixture of Gaussians in d dimensions

    Args:
        conf: The density_estimation configuration
        random: Service object encapsulating random behavior
        actual_A: The standard deviation matrix, the same one is applied across each Gaussian

    Return: A 1D numpy array of floating point numbers, drawn from the Gaussian mixture.
    """
    n = conf.n
    d = conf.d
    min_eigenvalue = conf.min_eigenvalue
    max_eigenvalue = conf.max_eigenvalue
    means = conf.means
    # Generate a standard deviation matrix A, by producing an eigen decomposition with a minimum limit for each
    # eigenvalue
    Q = _random_orthogonal_matrix(d)
    eigenvalues = random.uniform(min_eigenvalue, max_eigenvalue, d)
    upper_lambda = np.eye(d) * eigenvalues
    actual_A = np.matmul(Q, np.matmul(upper_lambda, Q.transpose()))

    chosen_means = random.choice(means, n, replace=True)

    z = random.normal_numpy_array([n, d])
    return chosen_means + np.matmul(z, actual_A), actual_A



def _random_orthogonal_matrix(d):
    H = np.eye(d)
    D = np.ones((d,))
    for n in range(1, d):
        x = np.random.normal(size=(d - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(d - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(d)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (d % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    Q = (D * H.T).T
    return Q