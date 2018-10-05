import numpy as np
import tensorflow as tf
import src.network_runner as nr

class ExperimentTrainer:
    """
    Class specifically designed to train experiments, i.e. synthetic data sets
    """

    def __init__(self, num_runs, epochs, batch_size):
        self.num_runs = num_runs
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, sess, network, training_set, validation_set, adversarial_set, run_number):
        """
        Train a newly initiated network on training_set, reporting accuarcy on validation_set along the way

        Args:
            network: A neural network
            training_set: The training set for the experiment at hand
            validation_set: The validation set for the experiment at hand

        Returns:
            A Network runner, an object that encapsulates the network and session of the trained network,
            for convenient post training analysis
        """
        X = training_set.X
        Y = training_set.Y
        n = X.shape[0]
        X_val = validation_set.X
        Y_val = validation_set.Y
        n_val = X_val.shape[0]

        train_indicies = np.arange(n)

        # Training
        for e in xrange(self.epochs):
            np.random.shuffle(train_indicies)
            for b in xrange(0, n, self.batch_size):
                batch = train_indicies[b:b + self.batch_size]
                x = X[batch]
                y = Y[batch]
                feed_dict = {network.x: x, network.y: y}
                _, a1, a, bias = sess.run((network.train_op, network.a1, network.a, network.b), feed_dict=feed_dict)
            if self.num_runs == 1:
                self._report_success(e, network, sess, training_set, validation_set, adversarial_set, 'Epoch')

        return self._report_success(run_number, network, sess, training_set, validation_set, adversarial_set)


    def _report_success(self, when, network, sess, ts, vs, advs, period_name='Run'):
        print '{period_name}: {e}'.format(period_name=period_name, e=when)
        train_all_correct = self._report_acc('Train', network, sess, ts)
        val_all_correct = self._report_acc('Val', network, sess, vs)
        adv_all_correct = self._report_acc('Adv', network, sess, advs)
        print ''
        return train_all_correct, val_all_correct, adv_all_correct

    def _report_acc(self, name, network, sess, data_set):
        X = data_set.X
        Y = data_set.Y
        n = X.shape[0]
        a = sess.run(network.a, feed_dict={network.x: X})
        num_correct = np.sum((np.argmax(a, axis=1) == Y).astype(dtype=np.int32))
        print '{name}: {num_correct} / {n}'.format(name=name, num_correct=num_correct, n=n)
        return num_correct == n
