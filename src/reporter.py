from visualisation import *
import numpy as np
from shortest_point_finder import find_shortest_point
from prediction_analytics import *
from adverserial import *

def report_single_network(network_runner, data_set, conf):
    """Module responsible for reporting on the results of a trained network.

    This includes analysis of the rbf parameters, an examination of the
    properties of correct, incorrect and adverserial examples etc."""
    X_val = data_set.X_val
    Y_val = data_set.Y_val
    correct, incorrect = _report(network_runner, data_set, conf)

    if conf.print_rbf_batch:
        _print_rbf_bacth(X_val, Y_val, network_runner, conf)

    class_to_adversary = conf.class_to_adversary_class
    if class_to_adversary is None:
        z, z_bar, tau = network_runner.report_rbf_params(X_val, Y_val)
        class_to_adversary = _report_shortest_point(conf, z_bar, tau)

    # show adversaries
    if conf.show_adversaries:
        x_adv, y_actual, x_actual = adverserial_gd(network_runner, correct, class_to_adversary, conf)
        adverse_prediction, _ = _get_probabilities_for(network_runner, x_adv, y_actual, report=True)
        plot_all_with_originals(x_adv, adverse_prediction, y_actual, x_actual)

    # Write structured data to a csv file
    if conf.write_csv:
        write_csv(X_val, Y_val, network_runner)

    if conf.show_roc:
        tprs, fprs = roc_curve(X_val, Y_val, network_runner, conf)
        visualisation.plot('ROC curve', fprs, tprs)

    # Show incorrect above the threshold
    if conf.show_really_incorrect:
        really_incorr_inds = np.argwhere(incorrect.prediction_prob() > conf.classified_as_thresh)[:, 0]
        really_incorrect_prediction = incorrect.prediction[really_incorr_inds]
        really_incorrect_x = incorrect.x[really_incorr_inds]
        really_incorrect_actual = incorrect.y[really_incorr_inds]
        print "Really Incorrect Actuals vs Predictions:"
        print 'A: '+str(really_incorrect_actual)
        print 'P: '+str(really_incorrect_prediction)
        plot_all(really_incorrect_x, really_incorrect_prediction, really_incorrect_actual)

def report_with_adverseries_from_second(nr1, nr2, data_set, conf):
    """Reporting function focused on generating adverser"""
    with nr1.graph.as_default():
        correct, incorrect, = _report(nr1, data_set, conf)
        class_to_adversary = conf.class_to_adversary_class
        if class_to_adversary is None:
            z, z_bar, tau = nr1.report_rbf_params(data_set.X_val, data_set.Y_val)
            class_to_adversary = _report_shortest_point(conf, z_bar, tau)

        x_adv, y_actual, x_actual = adverserial_gd(nr1, correct, class_to_adversary, conf)
        predictions1, probs1 = _get_probabilities_for(nr1, x_actual, y_actual)
        adv_predictions1, adv_probs1 = _get_probabilities_for(nr1, x_adv, y_actual)

    with nr2.graph.as_default():
        predictions2, probs2 = _get_probabilities_for(nr2, x_actual, y_actual)
        adv_predictions2, adv_probs2 = _get_probabilities_for(nr2, x_adv, y_actual)
        adv_ss = x_adv.shape[0]

    actual_class, adv_class = class_to_adversary
    _convincing_adverseries(adv_predictions1, adv_probs1, adv_class)
    _convincing_adverseries(adv_predictions2, adv_probs2, adv_class)

    # for i in xrange(adv_ss):
    #     print 'Actual: '+str(y_actual[i])
    #     _report_prediction_and_its_prob('Correct from this net: ', predictions1[i], probs1[i])
    #     _report_prediction_and_its_prob('Adversary from this net: ', adv_predictions1[i], adv_probs1[i])
    #     _report_prediction_and_its_prob('Correct from other net: ', predictions2[i], probs2[i])
    #     _report_prediction_and_its_prob('Adversary from other net: ', adv_predictions2[i], adv_probs2[i])
    #     print ''

    #plot_all_with_originals(x_adv, adv_predictions1, y_actual, x_actual)

def _convincing_adverseries(adv_predictions, adv_probs, adv_class):
    was_adverserial_prediction = adv_predictions == adv_class
    ss = adv_predictions.shape[0]
    exceeded_thresh = adv_probs[np.arange(ss), adv_predictions] > 0.5
    threating_adversary = np.logical_and(was_adverserial_prediction, exceeded_thresh).astype(np.int32)
    print 'Number of convincing adverseries: ' + str(np.sum(threating_adversary)) + ' / ' + str(ss)



def _report(network_runner, data_set, conf):
    X_val = data_set.X_val
    Y_val = data_set.Y_val
    correct, incorrect, _, _ = network_runner.all_correct_incorrect(X_val, Y_val)

    # Report on a sample of correct and incorrect results
    correct_sample = correct.sample(10)
    incorrect_sample = incorrect.sample(10)
    correct_sample.show()
    incorrect_sample.show()

    return correct, incorrect

def _report_shortest_point(conf, z_bar, tau):
    # Apply the shortest point finder
    sp_z_list, Cs, rbfs, z_bar_pair, tau_pair, closest_classes = find_shortest_point(conf, z_bar, tau)
    print 'Shortest Point Distance:'
    print Cs
    print rbfs
    print closest_classes
    print ''
    return closest_classes

def _report_prediction_and_its_prob(title, prediction, probs):
    print title
    print "Prediction: "+str(prediction)+"   "+str(np.max(probs))

def _get_probabilities_for(network_runner, x, y, report=False):
    a = network_runner.network.a
    probabilities = network_runner.feed_and_run(x, y, a)
    prediction = np.argmax(probabilities, axis=1)
    if report:
        for i in xrange(x.shape[0]):
            print i+1
            print 'Actual: ' + str(y[i])
            print probabilities[i]
            print ''
    return prediction, probabilities

def _print_rbf_bacth(X, Y, network_runner, conf):
    """Print the rbf parameters from a random batch of the data set"""
    z, z_bar, tau = network_runner.report_rbf_params(X, Y, conf.m)
    print str(z)+'\n'
    print str(z_bar) + '\n'
    print str(tau) + '\n'
