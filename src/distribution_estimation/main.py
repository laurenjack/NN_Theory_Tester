import tensorflow as tf
import numpy as np

import configuration
import data_generator as dg
import kernel_density_estimator
import trainer as tr
import density_collector
import density_animator
import pdf_functions as pf
from src import random_behavior


def run():
    conf = configuration.get_configuration()

    # Train H
    graph_H = tf.Graph()
    with graph_H.as_default():
        pdf_functions = pf.PdfFunctions(conf)
        random = random_behavior.Random()
        trainer = tr.Tranier(conf, random)
        collector = density_collector.NullCollector()
        data_generator = dg.DataGenerator(conf, pdf_functions, random)
        x, actual_A = data_generator.sample_gaussian_mixture(conf.n)
        kde_H = kernel_density_estimator.KernelDensityEstimator(conf, pdf_functions, data_generator)
        # Tensorflow setup
        session = tf.InteractiveSession()
        trained_H_inverse = trainer.train_H(kde_H, session, x, collector)
        session.close()

    print '\nActual H:\n'
    print np.matmul(actual_A, actual_A)

    # # Train h
    # graph_h = tf.Graph()
    # with graph_h.as_default():
    #     pdf_functions = pf.PdfFunctions(conf)
    #     data_generator = dg.DataGenerator(conf, pdf_functions, random)
    #     kde_h = kernel_density_estimator.KernelDensityEstimator(conf, pdf_functions, data_generator)
    #     # Tensorflow setup
    #     session = tf.InteractiveSession()
    #     collector = density_collector.create_chi_squared_collector(conf, random, trained_H_inverse, x, data_generator,
    #                                                                pdf_functions, session)
    #     trained_h = trainer.train_h(kde_h, session, x, collector, trained_H_inverse)
    #     session.close()
    #     animator = density_animator.UnivariateAnimator(conf)
    #     animator.animate_density(*collector.results())

    # collector = density_collector.MeanSquaredErrorCollector(conf, random, x, actual_A)
    # kde_trainer.train(kde, conf, session, random, x, collector)

    # # TODO(Jack) code smell here, duplicate code, refactor
    # if conf.d == 1:
    #     # Create a collector for animation
    #     collector = density_collector.create_univariate_collector(conf, random, x, actual_A)
    #     animator = density_animator.UnivariateAnimator(conf)
    #     kde_trainer.train(kde, conf, session, random, x, collector)
    #     animator.animate_density(*collector.results())
    # elif conf.d == 2:
    #     # Create a collector for animation
    #     collector = density_collector.create_multivariate_collector(conf, random, x)
    #     animator = density_animator.TwoDAnimator(conf, actual_A)
    #     kde_trainer.train(kde, conf, session, random, x, collector)
    #     animator.animate_density(*collector.results())
    # else:
    #     collector = density_collector.NullCollector()
    #     kde_trainer.train(kde, conf, session, random, x, collector)


if __name__ == '__main__':
    run()