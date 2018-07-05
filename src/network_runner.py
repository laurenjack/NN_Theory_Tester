import numpy as np

class NetworkRunner:
    """Within the context of a tensorflow session, responsible
    for loading numpy data into a network and running network operations

    These include most of the networks tensorflow ops, but also additional
    operations such as reporting on accuracy"""

    def __init__(self, network, session):
        self.network = network
        self.sess = session

    def feed_and_run(self, x, y, op, batch=None):
        if batch is not None:
            x = x[batch]
            y = y[batch]
        batch_size = x.shape[0]
        feed_dict = {self.network.get_x(): x, self.network.get_y(): y, self.network.get_batch_size(): batch_size}
        result = self.sess.run(op, feed_dict=feed_dict)
        return result

    def report_rbf_params(self, X, Y):
        """Report the rbf parameters (z, z_bar and tau) for the data set X, Y"""
        z, z_bar, tau = self.feed_and_run(X, Y, self.network.rbf_params())
        return z, z_bar, tau



    def report_accuracy(self, set_name, batch_indicies, accuracy_ss, X, Y):
        acc_batch = _random_batch(batch_indicies, accuracy_ss)
        a = self.feed_and_run(X, Y, self.network.a, acc_batch)
        y = Y[acc_batch]
        acc = self._compute_accuracy(a, y)
        print set_name + " Accuracy: " + str(acc)
        return acc

    def all_correct_incorrect(self, X, Y):
        """Get The prediction report for all the correct, and all the incorrect predictions"""
        n = X.shape[0]
        a = self.feed_and_run(X, Y, self.network.a)
        preds = np.argmax(a, axis=1)
        # Find the correct predictions
        is_correct = np.equal(Y, preds)
        correct_inds = np.argwhere(is_correct)[:, 0]
        incorr_inds = np.argwhere(np.logical_not(is_correct))[:, 0]
        corr = PredictionReport("Correct", a[correct_inds], X[correct_inds], Y[correct_inds])
        incorr = PredictionReport("Incorrect", a[incorr_inds], X[incorr_inds], Y[incorr_inds])
        return corr, incorr, correct_inds, incorr_inds

    def sample_correct_incorrect(self, ss, X, Y):
        corr, incorr, _, _ = self.all_correct_incorrect(X, Y)
        return corr.sample(ss), incorr.sample(ss)

    def _compute_accuracy(self, a, y):
        ss = y.shape[0]
        prediction = np.argmax(a, axis=1)
        correct_indicator = np.equal(prediction, y).astype(np.int32)
        return float(np.sum(correct_indicator)) / float(ss)


class PredictionReport:

    def __init__(self, name, a, x, y):
        self.name = name
        self.a = a
        self.x = x
        self.y = y
        self.prediction = np.argmax(self.a, axis=1)

    def show(self):
        print "Name: "+str(self.name)
        ss = self.y.shape[0]
        for i in xrange(ss):
            print "Actual: "+str(self.y[i])
            print "Prediction: "+str(self.a[i])
            print ""
        print "\n"

    def prediction_prob(self):
        return self.a[np.arange(self.a.shape[0]), self.prediction]

    def get_sample_of_class(self, k, ss):
        inds_of_class = np.argwhere(self.y == k)[:, 0]
        num_k = inds_of_class.shape[0]
        ss = min(ss, num_k)
        inds_of_sample = _random_batch(inds_of_class, ss)
        return PredictionReport(self.name, self.a[inds_of_sample], self.x[inds_of_sample], self.y[inds_of_sample])


    def sample(self, sample_size):
        """Return a new prediction report, based on a random sample of this prediction report"""
        m = self.a.shape[0]
        ss = min(m, sample_size)
        inds = np.arange(m)
        r_inds = _random_batch(inds, ss)
        return PredictionReport(self.name, self.a[r_inds], self.x[r_inds], self.y[r_inds])


def _random_batch(batch_indicies, m):
    return np.random.choice(batch_indicies, size=m, replace=False)
