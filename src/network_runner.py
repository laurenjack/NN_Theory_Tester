import numpy as np
import tensorflow as tf


class NetworkRunner(object):
    """Responsible for loading numpy data into a network and finding the value of specific tensors, within the context
    of a tensorflow session.

    Attributes:
        network: The network to train/run via session.run().
        session: The tensorflow session.
        graph: The tensorflow Graph which the network is/is a part of.
        m: The batch_size as specified by the network configuration.
    """

    def __init__(self, network, session, m, graph=None):
        self.network = network
        self.sess = session
        self.m = m
        self.graph = graph
        if self.graph is None:
            self.graph = tf.get_default_graph()

    def feed_and_run(self, x, y, op, indices=None, lr=None, sample_size=None):
        """Feed a data set (examples x, targets y) to the underlying network and run the operation/operations specified
        by the tensor/tensors op.

        If lr (learning rate) is specified then True will be fed to the networks is_training place holder, and a
        training step will take place (unless op does not contain any training/update tensors). This function is
        essentially a convenience wrapper for session.run().

        Args:
            x: A numpy array of input examples - e.g. could have shape [n, num_pixels]
            y: A numpy array of target classes, must have shape [n], where n = x.shape[0]
            op: A tensor,  or a list of tensors, or a tuple of tensors to run.
            indices: The caller may provide an array of indices with shape [s] where 0 < s < n. This array must contain
            a subset of the indices 0 -> n-1. The ordering of these indices specifies  of the order in which x, and y
            are processed and split into batches. If indices is not provided, an in-order array of indices of shape [n]
            will be created, so x and y are processed in order.
            lr: The learning rate, the network will train if you specify this and won't otherwise.
            sample_size: If this is an integer such that 0 < sample_size < n, then this function will operate on a
            random sample from the data_set, instead of using all indices.
        """
        self._feed_and_return(x, y, op, indices, lr, sample_size)

    def feed_and_return(self, x, y, op, indices=None, lr=None, sample_size=None):
        """Same as feed_and_run, but returns the numpy arrays corresponding to the tensor specified by op

        Returns: The single numpy array or the list of numpy arrays for op - the tensor/s submitted to
        session.run().
        """
        result_list = self._feed_and_return(x, y, op, indices, lr, sample_size, [])
        transposed_results = self._transpose(result_list)

        # Preserve how session.run() returns null results by returning them, keeping the length of the concatenated
        # results equal to the length of op.
        concatenated_results = []
        for r in transposed_results:
            if r and len(r[0].shape) > 0:
                concatenated_results.append(np.concatenate(r))
            else:
                # If r does not have a result, keep the concatenate result as null, if it is a list of scalars, keep it
                # as a list of scalars (most likely aggregates).
                # TODO(Jack) bug for aggregates, you probably over-engineered this.
                concatenated_results.append(r)

        if len(concatenated_results) == 1:
            return concatenated_results[0]
        return concatenated_results

    def probabilities(self, x, y):
        """Get the probabilities of the networks class predictions, also know as a.

        Let n = x.shape[0]

        Args:
            x: A set of examples, e.g. a numpy array of shape [n, num_pixel]
            y: A set of targets corresponding to x, must have shape [n]

        Returns: An [n, num_class] numpy array of probabilities.
        """
        return self.feed_and_return(x, y, self.network.a)

    def report_accuracy(self, set_name, x, y, indices=None, accuracy_ss=None):
        """Compute and print the accuracy of the network, i.e. evaluate the percentage examples where f(x[i]) == y[i].

        Args:
            set_name: The name of the data set x, y, e.g. 'Training Set'.
            x: A set of examples, e.g. a numpy array of shape [n, num_pixel].
            y: A set of targets corresponding to x, must have shape [n].

        Returns: A scalar acc, 0 <= acc <=1, representing the percentage of correct examples.
        """
        if indices is None:
            indices = np.arange(x.shape[0])
        if accuracy_ss:
            batch = _random_batch(indices, accuracy_ss)
        else:
            batch = indices
        a = self.feed_and_return(x, y, self.network.a, batch, sample_size=accuracy_ss)
        y = y[batch]
        acc = self._compute_accuracy(a, y)
        print set_name + " Accuracy: " + str(acc)
        return acc

    def all_correct_incorrect(self, x, y):
        """Separate the correct and incorrect predictions for the data set x, y, by the network.

        Args:
            x: A set of examples, e.g. a numpy array of shape [n, num_pixel].
            y: A set of targets corresponding to x, must have shape [n].

        Returns: A PredictionReport for all the correct examples followed by a PredictionReport of all the incorrect
        examples.
        """
        a = self.probabilities(x, y)
        prediction = np.argmax(a, axis=1)
        is_correct = np.equal(y, prediction)
        correct_indices = np.argwhere(is_correct)[:, 0]
        incorrect_indices = np.argwhere(np.logical_not(is_correct))[:, 0]
        correct = _build_prediction_Report("Correct", x, y, a, correct_indices)
        incorrect = _build_prediction_Report("Incorrect", x, y, a, incorrect_indices)
        return correct, incorrect

    def sample_correct_incorrect(self, x, y, sample_size):
        """Return two PredictionReports, a sample of correct examples and a sample of incorrect examples, both of which
        are size sample_size.

        Args:
            x: A set of examples, e.g. a numpy array of shape [n, num_pixel].
            y: A set of targets corresponding to x, must have shape [n].

        Returns: A PredictionReport for a sample of correct examples followed by a PredictionReport for a sample of
        incorrect examples.
        """
        correct, incorrect = self.all_correct_incorrect(x, y)
        return correct.sample(sample_size), incorrect.sample(sample_size)

    def _feed_and_return(self, x, y, op, indices, lr, sample_size, init_result_list=None):
        """See documentation of feed_and_run and feed_and_return"""
        if indices is None:
            n = x.shape[0]
            indices = np.arange(n)
        else:
            n = indices.shape[0]

        if sample_size and 0 < sample_size < n:
            indices = _random_batch(indices, sample_size)

        feed_dict = {}
        if lr:
            feed_dict[self.network.lr] = lr
            feed_dict[self.network.is_training] = True
        else:
            feed_dict[self.network.is_training] = False

        for k in xrange(0, n, self.m):
            batch = indices[k:k + self.m]
            batch_size = batch.shape[0]
            x_batch = x[batch]
            y_batch = y[batch]
            feed_dict[self.network.x] = x_batch
            feed_dict[self.network.y] = y_batch
            feed_dict[self.network.batch_size] = batch_size
            result = self.sess.run(op, feed_dict=feed_dict)
            if init_result_list is not None:
                init_result_list.append(result)
        return init_result_list

    def _transpose(self, result_list):
        """Group the numpy array results relating to the same tensor, in the same list.

        Essentially, if feed_and_return was called with a list of 3 tensors, b times, then result_list is a [b, 3] list
        of lists. This operation transposes that list of list to have the shape [3, b]. Additionally, if one of the
        tensors was something like a training operation, which returns None from sess.run(), rather than keeping a
        list of b None's, this function will put a single None in the place of that list.
        """
        first_result = result_list[0]

        # The case where op is a list or tuple of tensors, [b, 3] -> [3, b]
        if isinstance(first_result, (list, tuple)):
            # A list of lists, where each inner list corresponds to one of the tensors
            transposed_results = []
            for array in first_result:
                if array is not None:
                    transposed_results.append([])
                else:
                    transposed_results.append(None)

            for result in result_list:
                num_result = len(result)
                for array, i in zip(result, xrange(num_result)):
                    if transposed_results[i] is not None:
                        transposed_results[i].append(array)
            return transposed_results

        # The case where op was a single tensor, so [b] becomes [1, b]
        return [None] if result_list[0] is None else [result_list]

    def _compute_accuracy(self, a, y):
        ss = y.shape[0]
        prediction = np.argmax(a, axis=1)
        correct_indicator = np.equal(prediction, y.astype(np.int32)).astype(np.int32)
        return float(np.sum(correct_indicator)) / float(ss)


class RbfNetworkRunner(NetworkRunner):
    """NetworkRunner with additional behavior specific to rbf-softmax networks.
    """

    def __init__(self, network, session, m, graph=None):
        super(RbfNetworkRunner, self).__init__(network, session, m, graph)

    def report_rbf_params(self, x, y, ss=None):
        """Report the rbf parameters (z, z_bar and tau) for the data set X, Y
        Optional sample size for choosing random sample of data set"""
        return self.feed_and_return(x, y, self.network.rbf_params(), random_sample_size=ss)


class PredictionReport:
    """Convenient way of grouping the predictions the network made on a particular subset of the data set.

    Let the number of examples in this subset be n_sub.

    Attributes:
        name: The logical name for the subset, e.g 'All Correct Predictions'.
        x: The input examples, e.g. could have shape [n_sub, num_pixel]
        y: The targets, must have shape [n_sub].
        a: The probabilities as produced by the softmax. Shape: [n_sub, num_class]
        indices: The indices this subset had in the original data set of length n.
        prediction: The network's prediction, must have shape [n_sub].
    """

    def __init__(self, name, x, y, a, indices):
        self.name = name
        self.a = a
        self.x = x
        self.y = y
        self.indices = indices
        self.prediction = np.argmax(self.a, axis=1)

    def show(self):
        """Print this subset of predictions.
        """
        print "Name: "+str(self.name)
        ss = self.y.shape[0]
        for i in xrange(ss):
            print "Actual: "+str(self.y[i])
            print "Prediction: "+str(self.a[i])
            print ""
        print "\n"

    def prediction_prob(self):
        """The prediction probabilities, i.e. the probability the softmax gave for it's prediction (the highest).

        Returns: A numpy array of n prediction probabilities.
        """
        return self.a[np.arange(self.a.shape[0]), self.prediction]

    def get_sample_of_class(self, k, sample_size):
        """Take a sample of a single class k, and produce a new prediction report using just a sample of predictions
        that were of actual class k.

        Note that if sample_size exceeds the number of examples of class k, this will return all examples of class
        k in this prediction report.

        Args:
            k: The class to sample.
            sample_size: The number of examples of class k to sample.


        Returns: PredictionReport for a subset of size sample_size (if possible) of just class k
        """
        indices_of_class = np.argwhere(self.y == k)[:, 0]
        num_k = indices_of_class.shape[0]
        sample_size = min(sample_size, num_k)
        indices_of_sample = _random_batch(indices_of_class, sample_size)
        return _build_prediction_Report(self.name+'{}'.format(k), self.x, self.y, self.a, indices_of_sample)

    def sample(self, sample_size):
        """Return a new prediction report, based on a random sample of this prediction report.

        Args:
             sample_size: The number of examples the new prediction report will have, (technically it will have
             min(sample_size, n) examples.

        Returns: PredictionReport for a subset of size sample_size (if possible)
        """
        m = self.a.shape[0]
        ss = min(m, sample_size)
        indices = np.arange(m)
        random_indices = _random_batch(indices, ss)
        return _build_prediction_Report(self.name, self.x, self.y, self.a, random_indices)


def build_network_runner(graph, network, m, is_rbf=False):
    """Factory method for building a network runner, starts the tensorflow session and initialises the networks
    variables.

    graph: The tensorflow Graph which the network is/ is a part of.
    network: The network to train / report on
    m: The standard batch size as specified b the configuration
    is_rbf: Will create an instance of RbfNetworkRunner if true.

    Returns: An instance of NetworkRunner of one of it's subclasses.
    """
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    if is_rbf:
        return RbfNetworkRunner(network, sess, m, graph)
    return NetworkRunner(network, sess, m, graph)


def _build_prediction_Report(name, x, y, a, indices):
    """Constructs a PredictionReport, based on a subset of the data set x, y and its softmax probabilities a. This
    subset is specified by indices.

    Args:
        name: The name for the PredictionReport, e.g 'All Correct Predictions'.
        x: The input examples, e.g. could have shape [n, num_pixel]
        y: The targets, must have shape [n].
        a: The probabilities as produced by the softmax, must have shape [n, num_class].
        indices: A numpy array of indices of shape [n_sub] where n_sub <= n, which specifies the subset.
    """
    return PredictionReport(name, x[indices], y[indices], a[indices], indices)


def _random_batch(batch_indicies, m):
    return np.random.choice(batch_indicies, size=m, replace=False)

