import tensorflow as tf

import configuration
import kernel_density_estimator
import kde_trainer
from src import random_behavior


def run():
    conf = configuration.get_configuration()
    random = random_behavior.Random()

    # Initialise the distribution fitter
    kde = kernel_density_estimator.KernelDensityEstimator(conf)

    # Tensorflow setup
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    kde_trainer.train(kde, conf, sess, random)


if __name__ == '__main__':
    run()