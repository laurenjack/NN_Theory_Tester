import unittest
import mock
import numpy as np
import tensorflow as tf

from test import test_utils
from src.distribution_estimation import kernel_density_estimator


class TestKernelDensityEstimator(unittest.TestCase):

    def test_cost_is_correct(self):
        """ Test the Squared weighted mean error is correct. Manual working shown in comments
        """
        a_star = np.array([-1.0, 0.0, 1.0])
        a = np.array([-2.0, 1.0])

        # (a - a*[0]) = [-1.0, 2.0]
        # (a - a*[1]) = [-2.0, 1.0]
        # (a - a*[2]) = [-3.0, 0.0]

        # 2h^2 = 2 * 2^2 = 8
        # k = 1 / r = 1 / 3 (with h or 2 as its cancelled by the kernel)

        # f(a[0]) = k * (e ^ (-1 / 8) + e ^ (-4 / 8) + e(-9 / 8)) = 0.60456000988
        # f(a[1]) = k * (e ^ (-4 / 8) + e ^ (-1 / 8) + e ^ 0) = 0.8296758541

        # d00 = (a[0] - a*[0]) / f(a[0]) = -1.65409551351
        # d01 = (a[0] - a*[1]) / f(a[0]) = -3.30819102702
        # d02 = (a[0] - a*[2]) / f(a[0]) = -4.96228654053
        # d10 = (a[1] - a*[0]) / f(a[1]) = 2.41057997544
        # d11 = (a[1] - a*[1]) / f(a[1]) = 1.20528998772
        # d12 = (a[1] - a*[2]) / f(a[1]) = 0.0

        # Ed00 = e^(-1 / 8) * -1.65409551351 = -1.45973416725
        # Ed01 = e^(-4 / 8) * -3.30819102702 = -2.00651928607
        # Ed02 = e^(-9 / 8) * -4.96228654053 = -1.61101856912
        # Ed10 = e^(-4 / 8) * 2.41057997544 = 1.46209066279
        # Ed11 = e^(-1 / 8) * 1.20528998772 = 1.06366468088
        # Ed12 = 0.0

        # Mean over Edi0 = (-1.45973416725 + 1.46209066279) / 2.0 = 0.00117824777
        # Mean over Edi1 = (-2.00651928607 + 1.06366468088) / 2.0 = -0.47142730259
        # Mean over Edi2 = (-1.61101856912) / 2.0 = -0.80550928456

        # Squared weighted mean error = 0.5 * (0.64884520751 + 0.00117824777^2 + 0.47142730259 ^ 2) = 0.4355451487

        # Mock configuration - with essential params
        class ConfMock:
            def __init__(self):
                self.h_init = 2.0
                self.m = 2
                self.r = 3
                self.lr = 999.999  # irrelevant for this test
                self.float_precision = tf.float64  # High precision for testing
        conf = ConfMock()

        # Initialise the distribution fitter
        kde = kernel_density_estimator.KernelDensityEstimator(conf)

        # Tensorflow setup
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        _, squared_weighted_mean_error_op, _, _ = kde.squared_weighted_mean_error()
        swme = sess.run(squared_weighted_mean_error_op, feed_dict={kde.a: a, kde.a_star: a_star})

        self.assertAlmostEqual(0.4355451487, swme, places=7)