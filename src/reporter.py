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

    def report_single_network(self, network_runner, data_set):
        """Module responsible for reporting on the results of a trained network.

        This includes analysis of the rbf parameters, an examination of the
        properties of correct, incorrect and adversarial examples etc."""
        X_val = data_set.X_val
        Y_val = data_set.Y_val
        correct, incorrect = self._report(network_runner, data_set)

        if conf.print_rbf_params and conf.is_rbf:
            self._print_rbf_bacth(X_val, Y_val, network_runner)

        class_to_adversary = conf.class_to_adversary_class
        if class_to_adversary is None:
            z, z_bar, tau = network_runner.report_rbf_params(X_val, Y_val)
            class_to_adversary = self._report_shortest_point(z_bar, tau)

        # show adversaries
        if conf.show_adversaries:
            x_adv, y_actual, x_actual = adversarial_gd(network_runner, correct, class_to_adversary)
            adverse_prediction, _ = self._get_probabilities_for(network_runner, x_adv, y_actual, report=True)
            plot_all_with_originals(x_adv, adverse_prediction, y_actual, x_actual)

        # Write structured data to a csv file
        if conf.write_csv:
            self.prediction_analytics.write_csv(X_val, Y_val, network_runner)

        if conf.show_animation and conf.is_rbf:
            animate(*network_runner.ops_over_time)

        if conf.show_z_stats and conf.is_rbf:
            X_train = data_set.X_train
            Y_train = data_set.Y_train
            # correct, incorrect, _, _ = network_runner.all_correct_incorrect(X_train, Y_train)
            x_val_corr = correct.x
            y_val_corr = correct.y
            prediction_prob_corr = correct.prediction_prob()
            x_val_inc = incorrect.x
            y_val_inc = incorrect.y
            prediction_prob_inc = incorrect.prediction_prob()

            num_class = self.num_class
            z, z_bar, tau = network_runner.report_rbf_params(x_val_corr, y_val_corr)
            z_inc, _, _ = network_runner.report_rbf_params(x_val_inc, y_val_inc)
            for k in xrange(num_class):
                inds_of_k = np.argwhere(np.logical_and(y_val_corr == k, True))[:, 0]  # prediction_prob_corr > 0.6
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
            tprs, fprs = self.prediction_analytics.roc_curve(X_val, Y_val, network_runner)
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

    def report_with_adverseries_from_second(self, nr1, nr2, data_set):
        """Reporting function focused on generating adverser"""
        with nr1.graph.as_default():
            correct, incorrect, = self._report(nr1, data_set)
            class_to_adversary = conf.class_to_adversary_class
            if class_to_adversary is None:
                z, z_bar, tau = nr1.report_rbf_params(data_set.X_val, data_set.Y_val)
                class_to_adversary = self._report_shortest_point(z_bar, tau)

            x_adv, y_actual, x_actual = adversarial_gd(nr1, correct, class_to_adversary)
            predictions1, probs1 = self._get_probabilities_for(nr1, x_actual, y_actual)
            adv_predictions1, adv_probs1 = self._get_probabilities_for(nr1, x_adv, y_actual)

        with nr2.graph.as_default():
            predictions2, probs2 = self._get_probabilities_for(nr2, x_actual, y_actual)
            adv_predictions2, adv_probs2 = self._get_probabilities_for(nr2, x_adv, y_actual)
            adv_ss = x_adv.shape[0]

        actual_class, adv_class = class_to_adversary
        self._convincing_adverseries(adv_predictions1, adv_probs1, adv_class)
        self._convincing_adverseries(adv_predictions2, adv_probs2, adv_class)

        # for i in xrange(adv_ss):
        #     print 'Actual: '+str(y_actual[i])
        #     _report_prediction_and_its_prob('Correct from this net: ', predictions1[i], probs1[i])
        #     _report_prediction_and_its_prob('Adversary from this net: ', adv_predictions1[i], adv_probs1[i])
        #     _report_prediction_and_its_prob('Correct from other net: ', predictions2[i], probs2[i])
        #     _report_prediction_and_its_prob('Adversary from other net: ', adv_predictions2[i], adv_probs2[i])
        #     print ''

        # plot_all_with_originals(x_adv, adv_predictions1, y_actual, x_actual)

    def _convincing_adverseries(self, adv_predictions, adv_probs, adv_class):
        was_adversarial_prediction = adv_predictions == adv_class
        ss = adv_predictions.shape[0]
        exceeded_thresh = adv_probs[np.arange(ss), adv_predictions] > 0.5
        threating_adversary = np.logical_and(was_adversarial_prediction, exceeded_thresh).astype(np.int32)
        print 'Number of convincing adverseries: ' + str(np.sum(threating_adversary)) + ' / ' + str(ss)

    def _report(self, network_runner, data_set):
        X_val = data_set.X_val
        Y_val = data_set.Y_val
        correct, incorrect = network_runner.all_correct_incorrect(X_val, Y_val)

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
        probabilities = network_runner.probabilities()
        prediction = np.argmax(probabilities, axis=1)
        if report:
            for i in xrange(x.shape[0]):
                print i + 1
                print 'Actual: ' + str(y[i])
                print probabilities[i]
                print ''
        return prediction, probabilities

    def _print_rbf_bacth(self, X, Y, network_runner):
        """Print the rbf parameters from a random batch of the data set"""
        z, z_bar, tau = network_runner.report_rbf_params(X, Y, conf.m)
        print str(z) + '\n'
        print str(z_bar) + '\n'
        print str(tau) + '\n'