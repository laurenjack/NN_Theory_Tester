import tensorflow as tf
import numpy as np

from src.distribution_estimation import distribution_configuration
from src.distribution_estimation import pdf_functions as pf
from src.distribution_estimation import data_generator as dg
from src.distribution_estimation import kernel_density_estimator
from src.distribution_estimation import trainer as tr
from src.distribution_estimation import density_collector
from src import random_behavior


def run(conf, T):
    """For the distribution and bandwidth estimation algorithm specified by conf, generate a fresh set of data and
    train it with the algorithm T times.

    Args:
        data_generator: The generator to draw data from.
        T: The number of times to run the experiment

    Returns:
        E(h), Var(h) - A tuple, the sample mean and variance of the estimate.
    """
    # Construct the required services
    pdf_functions = pf.PdfFunctions()
    random = random_behavior.Random()
    gaussian_mixture = dg.GaussianMixture(conf, pdf_functions, random)
    kde = kernel_density_estimator.KernelDensityEstimator(conf, pdf_functions, gaussian_mixture)
    kde_tensors = kde.construct_kde_training_graph()
    trainer = tr.Tranier(conf, random)
    collector = density_collector.NullCollector()

    h_sum = 0.0
    sum_of_h_squares = 0.0
    session = tf.InteractiveSession()
    for t in xrange(T):
        x = gaussian_mixture.sample(conf.n)
        tf.global_variables_initializer().run()
        trained_h = trainer.train_R_for_gaussian_kernel(kde, kde_tensors, session, x, collector)
        h_sum += trained_h[0, 0]
        sum_of_h_squares += trained_h ** 2.0
        print "Run {}: {}".format(t + 1, trained_h)
    session.close()
    mean_h = h_sum / T
    variance_h = sum_of_h_squares / T - mean_h ** 2.0
    print '\nSample E(h) {}'.format(mean_h)
    print 'Sample Var(h) {}'.format(variance_h)
    return mean_h, variance_h