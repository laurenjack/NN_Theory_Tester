import unittest as test
import tensorflow as tf
import numpy as np
import main
from rbf import *

class TestRbf(test.TestCase):

    def test_gradients(self):
        conf = main.get_conf()

        conf.n = 3
        conf.num_class = 2
        conf.d = 2
        conf.rbf_c = 4.0
        conf.lr = 1.0

        z_init = tf.constant_initializer(np.array([[-1.5, 0.0], [1.0, -2.5], [2.5, 3.0]]))
        z_bar_init = tf.constant_initializer(np.array([[-1.0, 1.0],[-2.0, 1.0]]))
        tau_init = tf.constant_initializer(np.array([[-1.0, 0.5], [2.0, -3.0]]))
        y = np.array([0, 0, 1])
        rbf = RBF(conf, z_init, z_bar_init, tau_init)
        test_ops = rbf.test_ops()
        variable_ops = rbf.variable_ops()

        #init session
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        _, z_diff_sq, tau_sq, wxds, wxdso, norm_tau, tau, final_grad, variance_grad, probs, z_grad, z_bar_grad = sess.run(test_ops, feed_dict={rbf.y: y})
        final_z, final_z_bar, final_tau = sess.run(variable_ops, feed_dict={rbf.y: y})

        #Expectations
        exp_z_diff_sq = np.array([[[0.25, 6.25], [4.0, 1.0]], [[4.0, 0.0], [0.25, 12.25]], [[12.25, 2.25], [25.0, 4.0]]])
        exp_tau_sq = np.array([[1.0, 0.25], [4.0, 9.0]])
        exp_wxds = np.array([[[0.25, 1.5625], [16.0, 9]], [[4.0, 0.0], [1, 110.25]], [[12.25, 0.5625], [100, 36]]])
        exp_norm_tau = np.array([[1.265625, 0.19140625], [56.25, 1225]])
        exp_final_tau = np.array([[1.30290341, 0.58167315], [14.85320282, 66.69439697]])

        exp_probs = np.array([[ 0.49521008, 0.50478989], [0.58135545, 0.41864461], [0.5, 0.5]])
        exp_z_grad = np.array([[[-0.04453066, 0.0], [0.7124905, 0.0]],
                               [[0.41864455, 0.0], [-0.41864455, 0.0]],
                               [[0.0, 0.0147282 ], [0.0, 0.70695335]]])

        exp_z_bar_grad = np.array([[[0.03148793, -0.03497063], [-0.50380683, -0.50357711]],
                                   [[-0.29602644, 0.0], [0.29602644, -0.41864458]],
                                   [[0.08619016, -0.01041441], [0.49251521, -0.49989152]]], dtype=np.float32)


        #print z_bar_grad

        self.assertTrue(np.allclose(exp_z_diff_sq, z_diff_sq))
        self.assertTrue(np.allclose(exp_tau_sq, tau_sq))
        self.assertTrue(np.allclose(exp_wxds, wxds))
        self.assertTrue(np.allclose(exp_wxds, wxdso))
        self.assertTrue(np.allclose(exp_norm_tau, norm_tau))
        self.assertTrue(np.allclose(exp_final_tau, final_tau))

        self.assertTrue(np.allclose(exp_probs, probs))
        self.assertTrue(np.allclose(exp_z_grad, z_grad))
        self.assertTrue(np.allclose(exp_z_grad, z_grad))

        self.assertTrue(np.allclose(exp_z_bar_grad, z_bar_grad))

        # [[[-0.03148793  0.03497063]
        #   [0.50380683  0.50357711]]
        #
        # [[0.29602644 - 0.]
        # [-0.29602644
        # 0.41864458]]
        #
        # [[-0.08619016  0.01041441]
        #  [-0.49251521  0.49989152]]]

        # [[[0.06237829  0.0678985]
        #   [-0.9980526   0.99769223]]
        #
        # [[-0.70710677 - 0.]
        # [0.70710677
        # 1.]]
        #
        # [[-0.17238034 - 0.02082881]
        #  [-0.98503047 - 0.99978304]]]



if __name__ == '__main__':
    test.main()