import numpy as np


class RbfCollector(object):
    """A class responsible for collecting parameters and network outputs after each epoch of training, as training
    progresses.

    Attributes:
        class_wise_z_list: An [epochs, num_class] list of lists, representing the sampled z values which we are
        tracking over training.

        z_indices: The indices of our sampled z in the unshuffled training set, so we can keep track of where each
        sampled z is stored.

        z_bars: A list of [d, num_class] z_bar numpy matrices over time, one for each epoch once training is complete.

        taus: A list of [d, num_class] tau matrices over time, one for each epoch once training is complete.
    """

    def __init__(self, class_wise_z_list, z_indices):
        self.class_wise_z_list = class_wise_z_list
        self.z_indices = z_indices
        self.z_bars = []
        self.taus = []

    def collect(self, network_runner, x, y):
        """Collects the rbf parameters of the network, at the current epoch of training.

        This method must be called per epoch throughout training.

        Args:
            network_runner: A NetworkRunner instance for an rbf-softmax network.
        """
        network = network_runner.network
        num_class = len(self.class_wise_z_list)
        rbf_parameters = network.rbf_params()

        z, z_bar, tau = network_runner.feed_and_return(x, y, rbf_parameters, self.z_indices)
        for k in xrange(num_class):
            ind_of_class = np.argwhere(y[self.z_indices] == k)[:, 0]
            self.class_wise_z_list[k].append(z[ind_of_class])
        self.z_bars.append(z_bar)
        self.taus.append(tau)

    def results(self):
        """Returns: The three lists of rbf parameters, collected over training.
        """
        return self.class_wise_z_list, self.z_bars, self.taus



class NullCollector(object):
    """Following the null object pattern, used when there is no information that needs to be collected during training.
    """

    def __init__(self):
        pass

    def collect(self, network_runner, x, y):
        pass

    def results(self):
        return None


def build_rbf_collector(data_set, animation_ss):
    """Factory method for creating an RbfCollector.

    Args:
        data_set: The data set the rbf collector will report rbf parameters for (only effects z, not z_bar or tau)
        animation_ss: The Rbf collector is primarily for animating z instances over time, this integer parameter
        determines the sample size of z's to animate

    Returns: An instance of RbfCollector
    """
    num_class = data_set.num_class
    n = data_set.train.n
    train_indices = np.arange(n)
    z_indices = np.random.choice(train_indices, size=animation_ss, replace=False)
    class_wise_z_list = []
    for k in xrange(num_class):
        class_wise_z_list.append([])
    return RbfCollector(class_wise_z_list, z_indices)