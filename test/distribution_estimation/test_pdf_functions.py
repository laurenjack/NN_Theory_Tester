import unittest
import math
import numpy as np
import tensorflow as tf
from src.distribution_estimation import pdf_functions

Q = np.array([[1, 1], [1, -1]], dtype=np.float32) / 2 ** 0.5
lam_inv = np.array([0.5, 1], dtype=np.float32)
a = np.array([[1.0, 0.5]], dtype=np.float32)
a_star = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
means = np.array([[0.0, 0.0]], dtype=np.float32)


class PdfFunctionsSpec(unittest.TestCase):


    def test_sum_of_log_eigen_probs(self):
        result, _, _ = pdf_functions.sum_of_log_eigen_probs(Q, lam_inv, a, a_star, 1)
        dist0 = ((a[0] - a_star[0]).dot(Q) * lam_inv) ** 2
        dist1 = ((a[0] - a_star[1]).dot(Q) * lam_inv) ** 2
        f0 = np.exp(-0.5 * dist0) # / (2 * math.pi) ** 0.5 * lam_inv / 2.0
        f1 = np.exp(-0.5 * dist1) # / (2 * math.pi) ** 0.5 * lam_inv / 2.0
        f = f0 + f1
        expected = -np.log(2 * math.pi) - 2 * np.log(2) + np.sum(np.log(lam_inv)) + np.sum(np.log(f))
        # expected = np.sum(np.log(f))

        session = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        result = session.run(result)
        self.assertTrue(np.allclose(expected, result))


    def test_normal_exponent(self):
        A = np.matmul(Q / lam_inv, Q.transpose())
        si = np.linalg.inv(A.dot(A.transpose()))
        pa, log = pdf_functions.normal_exponent(a, means, Q, lam_inv, 1)
        session = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        result_pa, result_log = session.run([pa, log])

        x = a - means
        sigma_inverse = np.dot(Q * lam_inv ** 2.0, Q.transpose())
        distance = x.dot(sigma_inverse).dot(x.transpose())
        scale_comp = -np.log(2 * math.pi) + np.log(0.5)
        expected_log = -0.5 * distance + scale_comp
        expected_pa = np.exp(expected_log)

        self.assertTrue(np.allclose(expected_log, result_log))
        self.assertTrue(np.allclose(expected_pa, result_pa))

    def test_gradients_with_flex_weights(self):
        Q_tensor = tf.constant(Q)
        lam_inv_tensor = tf.constant(lam_inv)
        actual_Q = np.eye(2).astype(np.float32)
        actual_lamda_inverse = np.array([1.0, 3.0], dtype=np.float32)

        _, log_pa = pdf_functions.normal_exponent(a, means, actual_Q, actual_lamda_inverse, 1)
        log_fa, exponentials, difference = pdf_functions.sum_of_log_eigen_probs(Q_tensor, lam_inv_tensor, a, a_star, 1)
        weights = exponentials / tf.reshape(tf.reduce_sum(exponentials, axis=1), [1, 1, 2])
        log_delta = log_fa - log_pa

        # Create the actual gradients
        dQ, dlam_inv = pdf_functions.gradients_with_flex_weights(log_delta, difference, Q_tensor, lam_inv_tensor,
                                                                 weights, 1)
        # Now for MSE see what the gradients are and compare
        loss = tf.reduce_mean(log_delta ** 2) / 2
        dloss_dQ, d_loss_dlam_inv = tf.gradients(loss, [Q_tensor, lam_inv_tensor])

        session = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        dQ, dlam_inv, dloss_dQ, d_loss_dlam_inv = session.run([dQ, dlam_inv, dloss_dQ, d_loss_dlam_inv])
        self.assertTrue(np.allclose(dQ, dloss_dQ))
        self.assertTrue(np.allclose(dlam_inv, d_loss_dlam_inv))

