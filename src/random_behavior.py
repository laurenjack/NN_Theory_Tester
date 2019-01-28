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