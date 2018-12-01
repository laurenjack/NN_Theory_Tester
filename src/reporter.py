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

    def __init__(self, num_class, prediction_analytics):
        self.num_class = num_class
        self.prediction_analytics = prediction_analytics

    def report_single_network(self, network_runner, data_set, training_results=None):
        """Module responsible for reporting on the results of a trained network.

        This includes analysis of the rbf parameters, an examination of the
        properties of correct, incorrect and adversarial examples etc."""
        x_validation = data_set.validation.x
        y_validation = data_set.validation.y
        correct, incorrect = self._report(network_runner, data_set)

        if conf.print_rbf_params and conf.is_rbf:
            self._print_rbf_bacth(x_validation, y_validation, network_runner)

        class_to_adversary = conf.class_to_adversary_class
        if class_to_adversary is None:
            z, z_bar, tau = network_runner.report_rbf_params(x_validation, y_validation)
            class_to_adversary = self._report_shortest_point(z_bar, tau)

        # show adversaries
        if conf.show_adversaries:
            x_adv, y_actual, x_actual = adversarial_gd(network_runner, correct, class_to_adversary)
            adverse_prediction, _ = self._get_probabilities_for(network_runner, x_adv, y_actual, report=True)
            plot_all_with_originals(x_adv, adverse_prediction, y_actual, x_actual)

        # Write structured data to a csv file
        if conf.write_csv:
            self.prediction_analytics.write_csv(x_validation, y_validation, network_runner)

        if conf.show_animation and training_results:
            animate(*training_results)

        if conf.show_z_stats and conf.is_rbf:
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
            correct, incorrect, = self._report(first, data_set)
            class_to_adversary = conf.class_to_adversary_class
            if class_to_adversary is None:
                z, z_bar, tau = first.report_rbf_params(data_set.x_validation, data_set.y_validation)
                class_to_adversary = self._report_shortest_point(z_bar, tau)

            x_adv, y_actual, x_actual = adversarial_gd(first, correct, class_to_adversary)
            actual_predictions, actual_a = self._get_probabilities_for(first, x_actual, y_actual)
            adversarial_predictions, adversarial_a = self._get_probabilities_for(first, x_adv, y_actual)

        actual_class, adv_class = class_to_adversary
        print 'Correct examples evaluated on their source network:' # TODO(Jack) could remove, tautological
        self._report_number_convincing(actual_a, actual_predictions, actual_class, convincing_threshold)
        print 'Adversaries evaluated on their source network:'
        self._report_number_convincing(adversarial_a, adversarial_predictions, adv_class, convincing_threshold)


        adv_ss = x_adv.shape[0]
        successful_adv = adversarial_a[adversarial_predictions == adv_class]
        inds = np.arange(successful_adv.shape[0])
        print 'Successful Adv examples mean prediction probability of target: ' + str(
            np.mean(successful_adv[inds, adv_class]))
        plot_histogram(successful_adv[inds, adv_class])
        og = adversarial_a
        correct_inds = np.arange(actual_a.shape[0])
        print 'Correct examples mean prediction probability of target: ' + str(
            np.mean(actual_a[correct_inds, actual_class]))
        plot_histogram(actual_a[correct_inds, actual_class])

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

                con = adversarial_a[is_convincing_adversary]
                number_success = con.shape[0]
                if number_success > 0:
                    inds = np.arange(number_success)
                    print 'Convincing Adv examples mean prediction probability of target: ' + str(
                        np.mean(con[inds, adv_class]))
                    plot_histogram(con[inds, adv_class])

                cor = actual_a[is_convincing_correct]
                number_success = cor.shape[0]
                if number_success > 0:
                    inds = np.arange(number_success)
                    print 'Convincing Correct examples mean prediction probability of target: ' + str(
                        np.mean(cor[inds, actual_class]))
                    plot_histogram(cor[inds, actual_class])

        count_all_correct = self._count_true(correct_all_nets)
        print 'Correct all: {}'.format(count_all_correct)
        # Count how many adversaries from the first network fooled all subsequent networks
        count_fooled_all = self._count_true(attacked_all_nets)
        print 'Fooled all: {}'.format(count_fooled_all)

        fooled_all_a = og[attacked_all_nets]
        number_success = fooled_all_a.shape[0]
        if number_success > 0:
            inds = np.arange(number_success)
            print 'Successful Adv examples mean prediction probability of target: ' + str(
                np.mean(fooled_all_a[inds, adv_class]))
            plot_histogram(fooled_all_a[inds, adv_class])

        plot_all_with_originals(x_adv[attacked_all_nets], None, None, x_actual[attacked_all_nets])




        # for i in xrange(adv_ss):
        #     print 'Actual: '+str(y_actual[i])
        #     _report_prediction_and_its_prob('Correct from this net: ', predictions1[i], probs1[i])
        #     _report_prediction_and_its_prob('Adversary from this net: ', adv_predictions1[i], adv_probs1[i])
        #     _report_prediction_and_its_prob('Correct from other net: ', predictions2[i], probs2[i])
        #     _report_prediction_and_its_prob('Adversary from other net: ', adv_predictions2[i], adv_probs2[i])
        #     print ''

        # plot_all_with_originals(x_adv, adv_predictions1, y_actual, x_actual)

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

    def _count_true(self, boolean_array):
        return np.sum(boolean_array.astype(np.int32))

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