import numpy as np
import tensorflow as tf

from visualisation import *
from shortest_point_finder import find_shortest_point
from adversarial import *
from animator import animate
import configuration
conf = configuration.get_configuration()


class Reporter:
    """Module responsible for reporting on the results of a trained network

    Attributes:
        prediction_analytics: A service instance, see prediction_analytics.py
    """

    def __init__(self, num_class, prediction_analytics, real_conf):
        self.num_class = num_class
        self.prediction_analytics = prediction_analytics
        global conf
        conf = real_conf #TODO(Jack) Fix up

    def report_single_network(self, network_runner, data_set, training_results=None):
        """Module responsible for reporting on the results of a trained network.

        This includes analysis of the rbf parameters, an examination of the
        properties of correct, incorrect and adversarial examples etc."""
        x_validation = data_set.validation.x
        y_validation = data_set.validation.y
        correct, incorrect = self._report(network_runner, data_set)

        if conf.is_rbf and conf.print_rbf_params:
            self._print_rbf_bacth(x_validation, y_validation, network_runner)

        class_to_adversary = conf.class_to_adversary_class
        if class_to_adversary is None:
            z, z_bar, tau = network_runner.report_rbf_params(x_validation, y_validation)
            class_to_adversary = self._report_shortest_point(z_bar, tau)

        # show adversaries
        if conf.show_adversaries:
            x_adv, y_actual, x_actual = adversarial_gd(network_runner, correct, class_to_adversary, conf)
            adverse_prediction, _ = self._get_probabilities_for(network_runner, x_adv, y_actual, report=True)
            plot_all_with_originals(x_adv, adverse_prediction, y_actual, x_actual)

        # Write structured data to a csv file
        if conf.write_csv:
            self.prediction_analytics.write_csv(x_validation, y_validation, network_runner)

        if training_results and conf.show_animation:
            animate(*training_results)

        if conf.show_node_distributions:
            self._random_node_distributions(network_runner, x_validation, y_validation,
                                            conf.number_of_node_distributions)

        if conf.is_rbf and conf.show_z_stats:
            x_train = data_set.train.x
            y_train = data_set.train.y
            # correct, incorrect, _, _ = network_runner.all_correct_incorrect(x_train, y_train)
            x_validation_corr = correct.x
            y_validation_corr = correct.y
            prediction_prob_corr = correct.prediction_prob()
            x_validation_inc = incorrect.x
            y_validation_inc = incorrect.y
            prediction_prob_inc = incorrect.prediction_prob()

            num_class = self.num_class
            z, z_bar, tau = network_runner.report_rbf_params(x_validation_corr, y_validation_corr)
            z_inc, _, _ = network_runner.report_rbf_params(x_validation_inc, y_validation_inc)
            for k in xrange(num_class):
                inds_of_k = np.argwhere(np.logical_and(y_validation_corr == k, True))[:, 0]  # prediction_prob_corr > 0.6
                # length = inds_of_k.shape[0] // 2
                # s1 = inds_of_k[:length]
                # s2 = inds_of_k[length:]
                z_of_k = z[inds_of_k]
                z_mean = np.mean(z_of_k, axis=0)

                inds_of_k_inc = np.argwhere(np.logical_and(incorrect.prediction == k, True))[:,
                                0]  # prediction_prob_inc > 0.6
                z_of_k_inc = z_inc[inds_of_k_inc]
                z_mean_inc = np.mean(z_of_k_inc, axis=0)
                z_bar_k = z_bar[:, k].reshape(1, 4096)
                mean_diff = z_mean - z_mean_inc
                # mean_diff = np.mean(tau[:, k]**2.0 * (z_of_k_inc - z_bar_k)**2.0, axis=1) **0.5 #z_mean - prev_corr
                # sd = np.mean((z_of_k - z_mean.reshape(1, d)) ** 2.0, axis=0) ** 0.5
                # sd_tau_diff = sd - 1.0 / abs(tau[:, k]) * float(d)
                # mean_bar_diff = abs(tau[:, k]) * (z_bar[:, k] - z_mean)  #(abs(tau[:, k])*(z_bar[:, k] - z_mean)) ** 2.0
                if inds_of_k_inc.shape[0] > 0:
                    plot_histogram(mean_diff)
                    # plot_histogram(sd_tau_diff)
                    # inds_of_k = np.argwhere(y == k)[:, 0]
                    # plot_histogram(prediction_prob[inds_of_k])

        if conf.show_roc:
            tprs, fprs = self.prediction_analytics.roc_curve(x_validation, y_validation, network_runner)
            self.prediction_analytics.visualisation.plot('ROC curve', fprs, tprs)

        # Show incorrect above the threshold
        if conf.show_really_incorrect:
            prediction_prob = incorrect.prediction_prob()
            really_incorr_inds = np.argsort(-prediction_prob)[:conf.top_k_incorrect]
            really_incorrect_prediction = incorrect.prediction[really_incorr_inds]
            really_incorrect_prediction_probs = prediction_prob[really_incorr_inds]
            really_incorrect_x = incorrect.x[really_incorr_inds]
            really_incorrect_actual = incorrect.y[really_incorr_inds]
            print "Really Incorrect Actuals vs Predictions:"
            print 'A: ' + str(really_incorrect_actual)
            print 'P: ' + str(really_incorrect_prediction)
            print 'Prediction Probs: ' + str(really_incorrect_prediction_probs)
            plot_all_image(really_incorrect_x, really_incorrect_prediction, really_incorrect_actual)

    def report_with_adversaries_from_first(self, network_runners, data_set, convincing_threshold):
        """Reporting function focused on generating adverser"""
        first = network_runners[0]
        with first.graph.as_default():
            self._report_class_wise_accuracy(first, data_set)
            correct, incorrect, = self._report(first, data_set)
            class_to_adversary = conf.class_to_adversary_class
            if class_to_adversary is None:
                z, z_bar, tau = first.report_rbf_params(data_set.x_validation, data_set.y_validation)
                class_to_adversary = self._report_shortest_point(z_bar, tau)

            x_adv, y_actual, x_actual = adversarial_gd(first, correct, class_to_adversary)
            actual_predictions, actual_a = self._get_probabilities_for(first, x_actual, y_actual)
            adversarial_predictions, adversarial_a = self._get_probabilities_for(first, x_adv, y_actual)

        actual_class, adv_class = class_to_adversary
        print 'Correct examples evaluated on their source network:'
        isc = self._report_number_convincing(actual_a, actual_predictions, actual_class, convincing_threshold)
        print 'Adversaries evaluated on their source network:'
        isa = self._report_number_convincing(adversarial_a, adversarial_predictions, adv_class, convincing_threshold)

        # print 'Correct Activation Distribution'
        # self._activation_mean_and_histogram(actual_a, isc, actual_class)
        # print 'Adversarial Activation Distribution'
        # self._activation_mean_and_histogram(adversarial_a, isa, adv_class)

        adv_ss = x_adv.shape[0]
        attacked_all_nets = np.ones(adv_ss, dtype=np.bool)
        correct_all_nets = np.ones(adv_ss, dtype=np.bool)
        print 'Subsequent networks, what is the success of transferred attacks?'
        for network_runner in network_runners[1:]:
            with network_runner.graph.as_default():
                print '{} - Correct Transfer:'.format(network_runner.network.end.network_id)
                actual_predictions, actual_a = self._get_probabilities_for(network_runner, x_actual, y_actual)
                is_convincing_correct = self._report_number_convincing(actual_a, actual_predictions, actual_class,
                                                                       convincing_threshold)
                correct_all_nets = np.logical_and(correct_all_nets, is_convincing_correct)

                print 'Adversarial Transfer:'
                adversarial_predictions, adversarial_a = self._get_probabilities_for(network_runner, x_adv, y_actual)
                is_convincing_adversary = self._report_number_convincing(adversarial_a, adversarial_predictions,
                                                                         adv_class, convincing_threshold)
                attacked_all_nets = np.logical_and(attacked_all_nets, is_convincing_adversary)
                print ''
                # print 'Transferred Adversarial Activation Distribution'
                # self._activation_mean_and_histogram(adversarial_a, is_convincing_adversary, adv_class)

        count_all_correct = self._count_true(correct_all_nets)
        print 'Correct all: {}'.format(count_all_correct)
        # Count how many adversaries from the first network fooled all subsequent networks
        count_fooled_all = self._count_true(attacked_all_nets)
        print 'Fooled all: {}'.format(count_fooled_all)

        if conf.show_adversaries:
            plot_all_with_originals(x_adv[attacked_all_nets], None, None, x_actual[attacked_all_nets])

    def _report_number_convincing(self, a, predictions, target_class, convincing_threshold):
        """ Of a set of predictions on n examples, report how many of those examples exceeded the convincing threshold.

        Args:
            a: The output of the softmax, an [n, num_class] matrix of probabilities.
            predictions: An [n] shaped array of class predictions, each correspond to the row-wise max of a.
            target_class: The class of the example, in the adversarial case, this is the adversarial class we have
            targeted the example with.
            convincing_threshold: The threshold probability over which an example is considered convincing

        Returns:
            An [n] shaped boolean array, where true values indicate the example at index i exceeded the threshold.
        """
        hit_target = predictions == target_class
        n = predictions.shape[0]
        exceeded_thresh = a[np.arange(n), predictions] > convincing_threshold
        is_convincing = np.logical_and(hit_target, exceeded_thresh)
        number_of_convincing = self._count_true(is_convincing)
        print 'Number of Convincing: {} / {}'.format(number_of_convincing, n)
        return is_convincing

    def _activation_mean_and_histogram(self, a, is_convincing, target_class):
        """Print the mean and show a histogram of the activations all activations that were convincing incarnations
        of the target_class.

        Note that this method assumes that is_convincing was calculated using target_class.
        """
        convincing_a = a[is_convincing]
        number_success = convincing_a.shape[0]
        if number_success > 0:
            indices = np.arange(number_success)
            print 'Mean prediction probability of convincing examples: {}'. \
                format(np.mean(convincing_a[indices, target_class]))
            plot_histogram(convincing_a[indices, target_class])

    def _count_true(self, boolean_array):
        return np.sum(boolean_array.astype(np.int32))

    def _random_node_distributions(self, network_runner, x, y, number_of_node_distributions):
        """Choose number_of_node_distributions random nodes from an NN and plot their distributions.
        """
        network = network_runner.network
        activation_list = network.activation_list
        num_layer = len(activation_list)
        for i in xrange(number_of_node_distributions):
            self._choose_node(network_runner, x, y, activation_list, num_layer)

    def _choose_node(self, network_runner, x, y, activation_list, num_layer):
        # Choose a layer
        l = np.random.randint(num_layer)
        a = activation_list[l]
        a_transpose = tf.transpose(a)
        shape = a_transpose.shape
        indices_transpose = []
        for s in shape[:-1]:
            index = np.random.randint(s.value)
            indices_transpose.append(index)
        # Extract the activations from the single node
        single_node_transpose = tf.gather_nd(a_transpose, [indices_transpose])
        single_node = tf.transpose(single_node_transpose)
        indices = np.array(indices_transpose).transpose()
        activations = network_runner.feed_and_return(x, y, single_node)[:, 0]
        show_distribution(activations, l, indices)

    def _report(self, network_runner, data_set):
        x_validation = data_set.validation.x
        y_validation = data_set.validation.y
        correct, incorrect = network_runner.all_correct_incorrect(x_validation, y_validation)

        # Report on a sample of correct and incorrect results
        correct_sample = correct.sample(10)
        incorrect_sample = incorrect.sample(10)
        correct_sample.show()
        incorrect_sample.show()

        return correct, incorrect

    def _report_shortest_point(self, z_bar, tau):
        # Apply the shortest point finder
        sp_z_list, Cs, rbfs, z_bar_pair, tau_pair, closest_classes = find_shortest_point(z_bar, tau)
        print 'Shortest Point Distance:'
        print Cs
        print rbfs
        print closest_classes
        print ''
        return closest_classes

    def _report_prediction_and_its_prob(self, title, prediction, probs):
        print title
        print "Prediction: " + str(prediction) + "   " + str(np.max(probs))

    def _get_probabilities_for(self, network_runner, x, y, report=False):
        a = network_runner.network.a
        probabilities = network_runner.probabilities(x, y)
        prediction = np.argmax(probabilities, axis=1)
        if report:
            for i in xrange(x.shape[0]):
                print i + 1
                print 'Actual: ' + str(y[i])
                print probabilities[i]
                print ''
        return prediction, probabilities

    def _print_rbf_bacth(self, x, y, network_runner):
        """Print the rbf parameters from a random batch of the data set"""
        z, z_bar, tau = network_runner.report_rbf_params(x, y, conf.m)
        print str(z) + '\n'
        print str(z_bar) + '\n'
        print str(tau) + '\n'

    def _report_class_wise_accuracy(self, network_runner, data_set):
        x = data_set.train.x
        y = data_set.train.y
        for k in xrange(self.num_class):
            is_this_class = y == k
            x_k = x[is_this_class]
            y_k = y[is_this_class]
            class_name = 'Class {}'.format(k)
            network_runner.report_accuracy(class_name, x_k, y_k)