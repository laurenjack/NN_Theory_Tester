import numpy as np

class Random(object):
    """Encapsulates the random behavior across numpy and tensorflow.
    """

    def normal_numpy_array(self, shape):
        return np.random.randn(*shape)

    def shuffle(self, numpy_array):
        """Shuffle a numpy array
        """
        np.random.shuffle(numpy_array)

    def choice(self, x, ss, replace=False):
        """ Given a numpy array, take a random sample of ss, along it's first axis (without replacement by default).
        """
        # Chose from an array of indices, this allows to select from len(x.shape) > 1
        indices = np.arange(x.shape[0])
        selected_indices = np.random.choice(indices, size=ss, replace=replace)
        return x[selected_indices]