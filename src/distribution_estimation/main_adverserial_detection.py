from src import network_factory
from src import reporter_factory
from src import random_behavior
from src import adversarial
import configuration_network
import configuration
import pdf_functions as pf
import kernel_density_estimator
import trainer as tr
import tensorflow as tf
import density_collector
import numpy as np

def run():
    """Train a neural network and test distribution estimations as a means of detecting adverserial examples for that
    network.
    """
    conf_network = configuration_network.get_configuration()
    adversarial_ss = conf_network.adversarial_ss
    class_to_adversary = conf_network.class_to_adversary_class
    network_runner, data_set, training_results = network_factory.create_and_train_network(conf_network)

    train = data_set.train
    validation = data_set.validation
    # Get all the correct training points, these will form the basis of the activation distribution
    correct_train, _ = network_runner.all_correct_incorrect(train.x, train.y)
    activation_list_op = network_runner.network.activation_list
    a = network_runner.feed_and_return(correct_train.x, correct_train.y, activation_list_op)

    # Get the validation points, we will use these to test the detection algorithm
    correct_validation, _ = network_runner.all_correct_incorrect(validation.x, validation.y)
    x_adv, y_original, x_original = adversarial.adversarial_gd(network_runner, correct_validation, class_to_adversary,
                                                               conf_network)
    corr, incorr = network_runner.all_correct_incorrect(x_adv, y_original)
    print corr.indices
    a_original = network_runner.feed_and_return(x_original, y_original, activation_list_op)
    a_adv = network_runner.feed_and_return(x_adv, y_original, activation_list_op)

    # Train the chi square distance estimator
    conf_distribution_estimation = configuration.get_configuration()
    collector = density_collector.NullCollector()
    random = random_behavior.Random()
    trainer = tr.Tranier(conf_distribution_estimation, random)
    pdf_functions = pf.PdfFunctions(conf_distribution_estimation)
    kde = kernel_density_estimator.KernelDensityEstimator(conf_distribution_estimation, pdf_functions)
    session = tf.InteractiveSession()
    H_inverse = trainer.train_H(kde, session, np.copy(a), collector)

    # Compare the 10 closest points to the originals, and the adversaries
    def _report_closest_points(wd, x_batch, title):
        m = x_batch.shape[0]
        # exponent = exponent.transpose()
        # fa = fa.transpose()
        wd = wd.transpose()
        sorted_indices = np.argsort(wd, axis=1)
        top_10 = sorted_indices[:, 0:10]
        for i in xrange(wd.shape[0]):
            top_10_this_example = top_10[i]
            top_10_exponents = wd[i][top_10_this_example]
            top_10_x = x_batch[top_10_this_example]
            probs = network_runner.probabilities(top_10_x, np.zeros(10))
            predictions_of_neighbours = np.argmax(probs, axis=1)
            # top_10_fa = fa[i][top_10_indices]
            print 'Top n for '+title
            print top_10_exponents
            print predictions_of_neighbours
            # print top_10_fa
            print ''

    m = conf_distribution_estimation.m
    original_wd = pdf_functions._weighted_distance(H_inverse, a[0:m], a_original, m)
    adv_wd = pdf_functions._weighted_distance(H_inverse, a[0:m], a_adv, m)
    original_wd, adv_wd = session.run([original_wd, adv_wd])
    _report_closest_points(original_wd, correct_train.x[0:m], 'original')
    _report_closest_points(adv_wd, correct_train.x[0:m], 'adversarial')

    print corr.indices


    #reporter = reporter_factory.create_reporter(data_set, conf)
    #reporter.report_single_network(network_runner, data_set, training_results)


if __name__ == '__main__':
    run()