import tensorflow as tf

import configuration
import data_generator
import kernel_density_estimator
import kde_trainer
import density_collector
import density_animator
from src import random_behavior


def run():
    conf = configuration.get_configuration()
    random = random_behavior.Random()

    # Initialise the distribution fitter
    kde = kernel_density_estimator.KernelDensityEstimator(conf)

    # Tensorflow setup
    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    x = data_generator.generate_univariate_gaussian_mixture(conf, random)

    # Create a collector for animation
    collector = density_collector.create_univariate_collector(conf, random, x)

    animator = density_animator.Animator(conf)
    kde_trainer.train(kde, conf, session, random, x, collector)
    animator.animate_density(*collector.results())


if __name__ == '__main__':
    run()