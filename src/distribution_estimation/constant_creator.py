import numpy as np


def random_orthogonal_matrix(d):
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
    return Q.astype(np.float32)