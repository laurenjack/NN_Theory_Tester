import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from src.distribution_estimation import distribution_configuration
from src.distribution_estimation import pdf_functions as pf
from  src.distribution_estimation import data_generator as dg
from src.distribution_estimation import kernel_density_estimator
from src.distribution_estimation import trainer as tr
from src.distribution_estimation import density_collector
from src import random_behavior


def run(T):
    """Run the bandwidth estimation algorithm multiple times on a standard Gaussian distribution.

    Over these T runs Plot the average h for each epoch. Show this average h against the optimal h.
    """
    # Create a configuration based on a standard normal distribution
    conf = distribution_configuration.get_configuration()
    conf.d = 1
    conf.fixed_A = np.array([[1.0]], dtype=np.float32)
    conf.means = np.array([[0.0]], dtype=np.float32)
    # Training Parameters
    conf.fit_to_underlying_pdf = False
    conf.n = 10000
    conf.m = 100
    conf.r = 1000
    conf.R_init = 1.0 * np.eye(conf.d, dtype=np.float32)  # np.exp(-0.5) *
    conf.c = 0.2  # ** (1.0 / float(conf.d))
    conf.epochs = 100
    conf.lr_R = 0.05  # * (2 * math.pi * float(conf.d)) #** 0.5
    conf.reduce_lr_epochs = [24, 48, 72]
    # The factor to scale the learning rate down by
    conf.reduce_lr_factor = 0.3
    conf.show_variable_during_training = False
    conf.fit_to_underlying_pdf = False

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
        h_sum += trained_h[0,0]
        sum_of_h_squares += trained_h ** 2.0
        print "Run {}: {}".format(t + 1, trained_h)
    mean_h = h_sum / T
    variance_h = sum_of_h_squares / T - mean_h ** 2.0
    print '\nSample E(h) {}'.format(mean_h)
    print 'Sample Var(h) {}'.format(variance_h)
    session.close()


if __name__ == '__main__':
    run(30)