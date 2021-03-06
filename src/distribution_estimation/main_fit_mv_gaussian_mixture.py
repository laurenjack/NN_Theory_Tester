import tensorflow as tf
import numpy as np

import mv_kernel_configuration
from src import random_behavior
import data_generator as dg
import train_mv_kernel_estimator
import multivariate_kernel_service
import pdf_functions as pf


def run():
    """This script runs the bandwidth estimation algorithm, on an artificially generated mixture of Gaussians.

    It's purpose is validating that the bandwidth estimation algorithm works for Gaussian-mixture like distributions.
    The documentation of how both the Gaussian-mixture dataset and the algorithm are configured can be found in:
    mv_kernel_configuration.py. For a high level description of the algorithm and the corresponding mathematics see:
    TODO(Jack) put link in
    """
    # Construct the Multivariate Kernel densimeans, ty Estimator Graph - there's a bit of manual dependency injection here
    conf = mv_kernel_configuration.get_configuration()
    random = random_behavior.Random()
    data_generator = dg.GaussianMixture(conf, random)
    actuals = None
    if conf.fit_to_underlying_pdf:
        actuals = (data_generator.Q, data_generator.lam_inv, conf.means)
    distance_function = pf.EigenDistance()
    # distance_function = pf.IndependentDistance()
    pdf_service = pf.PdfFunctionService(distance_function)
    mv_service = multivariate_kernel_service.MultivariateKernelService(pdf_service, conf, actuals)
    trainer = train_mv_kernel_estimator.Trainer(conf, random)

    # Generate random samples from the data, our training process will use them to form f(a).
    x = data_generator.sample(conf.n)

    # Build graph and train
    trainer.train(mv_service, x, 1.0 / data_generator.lam_inv)
    print 'Actual Lamda: {}'.format(1.0 / data_generator.lam_inv)
    print 'Actual Q: {}'.format(data_generator.Q)


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    run()