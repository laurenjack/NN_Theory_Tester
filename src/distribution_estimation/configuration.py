conf = None  # singleton reference


def get_configuration():
    global conf
    if not conf:
        conf = Configuration()
    return conf


class Configuration:
    """Encapsulates the parameters regarding the structure, dataset, and training parameters of a Distribution
    Estimator.
    """

    def __init__(self):
        # The number of observations in the dataset.
        self.n = 1000
        # The number of training epochs
        self.epochs = 10
        # The number of examples for training at each step
        self.m = 100
        # The number of reference examples (those part of the Kernel density estimate) for each training step
        self.r = 100
        # The learning rate for h
        self.lr = 1.0

        self._validate()


    def _validate(self):
        """Validate there aren't any inconsistencies in the configuration (just looks out for non-obvious ones).
        """

        if self.m + self.r > self.n:
            raise ValueError('There are {n} examples in the data set but more ({m} + {r}) are being drawn.'
                             .format(n=self.n, m=self.m, r=self.r))
