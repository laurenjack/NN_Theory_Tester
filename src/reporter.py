import numpy as np
from shortest_point_finder import find_shortest_point
from prediction_output_writer import *
from adverserial import *

def report(network_runner, data_set, conf):
    """Module responsible for reporting on the results of a trained network.

    This includes analysis of the rbf parameters, an examination of the
    properties of correct, incorrect and adverserial examples etc."""
    X_val = data_set.X_val
    Y_val = data_set.Y_val
    z, z_bar, tau = network_runner.report_rbf_params(X_val, Y_val)
    correct, incorrect, _, _ = network_runner.all_correct_incorrect(X_val, Y_val)

    # Report on a sample of correct and incorrect results
    correct_sample = correct.sample(10)
    incorrect_sample = incorrect.sample(10)
    correct_sample.show()
    incorrect_sample.show()

    # Apply the shortest point finder
    sp_z_list, Cs, rbfs, z_bar_pair, tau_pair, closest_classes = find_shortest_point(conf, z_bar, tau)
    print 'Shortest Point Distance:'
    print Cs
    print rbfs
    print closest_classes

    # show adversaries
    adverserial_gd(network_runner, correct, closest_classes, conf)

    # Write structured data to a csv file
    write_csv(X_val, Y_val, network_runner)

    # Show incorrect above the threshold
    if conf.show_really_incorrect:
        really_incorr_inds = np.argwhere(incorrect.prediction_prob() > 0.5)[:, 0]
        really_incorrect_prediction = incorrect.prediction[really_incorr_inds]
        really_incorrect_x = incorrect.x[really_incorr_inds]
        really_incorrect_actual = incorrect.y[really_incorr_inds]
        plot_all(really_incorrect_x, really_incorrect_prediction, really_incorrect_actual)