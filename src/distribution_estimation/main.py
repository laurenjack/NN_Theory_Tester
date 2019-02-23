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

    x = data_generator.generate_gaussian_mixture(conf, random)

    # collector = density_collector.MeanSquaredErrorCollector(conf, random, x)
    # kde_trainer.train(kde, conf, session, random, x, collector)

    # TODO(Jack) code smell here, duplicate code, refactor
    if conf.d == 1:
        # Create a collector for animation
        collector = density_collector.create_univariate_collector(conf, random, x)
        animator = density_animator.UnivariateAnimator(conf)
        kde_trainer.train(kde, conf, session, random, x, collector)
        animator.animate_density(*collector.results())
    elif conf.d == 2:
        # Create a collector for animation
        collector = density_collector.create_multivariate_collector(conf, random, x)
        animator = density_animator.TwoDAnimator(conf)
        kde_trainer.train(kde, conf, session, random, x, collector)
        animator.animate_density(*collector.results())
    else:
        collector = density_collector.NullCollector()
        kde_trainer.train(kde, conf, session, random, x, collector)


if __name__ == '__main__':
    run()