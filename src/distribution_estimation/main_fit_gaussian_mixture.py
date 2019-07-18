import tensorflow as tf

import distribution_configuration
import data_generator as dg
import kernel_density_estimator
import trainer as tr
import density_collector
import density_animator
import pdf_functions as pf
from src import random_behavior


def run():
    """This script runs the bandwidth estimation algorithm, on an artificially generated mixture of Gaussians.

    It's purpose is validating that the bandwidth estimation algorithm works for Gaussian-mixture like distributions.
    The documentation of how both the Gaussian-mixture dataset and the algorithm are configured can be found in:
    distribution_configuration.py. For a high level description of the algorithm and the corresponding mathematics see:
    TODO(Jack) put link in
    """
    # Construct the Kernel density Estimator Graph - there's a bit of manual dependency injection here
    conf = distribution_configuration.get_configuration()
    pdf_functions = pf.PdfFunctions()
    random = random_behavior.Random()
    trainer = tr.Tranier(conf, random)
    data_generator = dg.GaussianMixture(conf, pdf_functions, random)
    kde = kernel_density_estimator.KernelDensityEstimator(conf, pdf_functions, data_generator)
    # Create the graph
    kde_tensors = kde.construct_kde_training_graph()

    # Generate random samples from the data, our training process will use them to form f(a).
    x = data_generator.sample(conf.n)

    # # Either set up a collector and animator, or don't collect anything where d > 2.
    # if conf.d == 1:
    #     collector = density_collector.create_univariate_collector(conf, random, x, actual_A)
    #     animator = density_animator.UnivariateAnimator(conf)
    # elif conf.d == 2:
    #     # Create a collector for animation
    #     collector = density_collector.create_multivariate_collector(conf, random, x)
    #     animator = density_animator.TwoDAnimator(conf, actual_A)
    # else:
    collector = density_collector.NullCollector()
    animator = density_animator.NullAnimator()

    # Graph created, data generated, collector setup - ready to actually train.
    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    trained_A_inverse = trainer.train_R_for_gaussian_kernel(kde, kde_tensors, session, x, collector)
    session.close()

    print '\nActual A:\n'
    print data_generator.actual_A

    # Animate the training process, if d ==1 or d == 2.
    animator.animate_density(collector)


if __name__ == '__main__':
    run()