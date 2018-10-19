import unittest as test
import tensorflow as tf
import numpy as np
import rbf as rb
import configuration

class TestRbf(test.TestCase):

    def test_gradients(self):
        conf = configuration.get_conf()

        conf.m = 3
        conf.d = 2
        conf.rbf_c = 4.0
        conf.lr = 1.0


        z_start = np.array([[-1.5, 0.0], [1.0, -2.5], [2.5, 3.0]], dtype=np.float32)
        z_bar_start = np.array([[-1.0, 1.0],[-2.0, 1.0]], dtype=np.float32)
        tau_start = np.array([[1.0, 0.5], [2.0, 3.0]])
        z_init = tf.constant_initializer(z_start)
        z_bar_init = tf.constant_initializer(z_bar_start)
        tau_init = tf.constant_initializer(tau_start)
        y = np.array([0, 0, 1])
        y_hot = np.array([[1, 0],[1, 0],[0, 1]]).reshape(3,1,2)
        z_var = tf.get_variable("z", shape=[conf.m, conf.d], initializer=z_init)
        net = rb.Rbf(z_bar_init, tau_init)
        rbf_ops = net.create_all_ops(z_var)

        # Get ops for testing
        train_op = rbf_ops.train_op
        z_bar_op = rbf_ops.z_bar
        tau_op = rbf_ops.tau
        z_diff_sq_op = rbf_ops.z_diff_sq
        tau_sq_op = rbf_ops.tau_sq
        wzds_op = rbf_ops.wzds
        wzdso_op = rbf_ops.wzds
        tau_grad_op = rbf_ops.tau_grad
        z_grad_op = rbf_ops.z_grad
        z_bar_grad_op = rbf_ops.z_bar_grad
        test_ops = [train_op, z_diff_sq_op, tau_sq_op, wzds_op, wzdso_op, tau_grad_op,
                    z_grad_op, z_bar_grad_op]
        var_ops = [z_var, z_bar_op, tau_op]

        #init session
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        feed_dict = {net.y: y, rb.batch_size: conf.m}
        _, z_diff_sq, tau_sq, wzds, wzdso, tau_grad, z_grad,\
        z_bar_grad = sess.run(test_ops, feed_dict=feed_dict)
        z, z_bar, tau = sess.run(var_ops, feed_dict=feed_dict)

        #Expectations
        exp_z_diff_sq = np.array([[[0.25, 6.25], [4.0, 1.0]], [[4.0, 0.0], [0.25, 12.25]], [[12.25, 2.25], [25.0, 4.0]]], dtype=np.float32)
        exp_tau_sq = np.array([[1.0, 0.25], [4.0, 9.0]], dtype=np.float32)
        exp_wzds = np.array([[[0.25, 1.5625], [16.0, 9]], [[4.0, 0.0], [1, 110.25]], [[12.25, 0.5625], [100, 36]]], dtype=np.float32)

        ss = np.array([2.0, 1.0], dtype=np.float32).reshape(1, 2)
        dC_dwzds = - 2.0 * (conf.target_precision - np.sum(y_hot * exp_wzds, axis=0) / ss)
        dwzds_dtau = 2.0 * tau_start * y_hot * exp_z_diff_sq / ss
        dC_dtau = dC_dwzds * dwzds_dtau
        exp_tau_grad = np.sign(dC_dtau) * abs(dC_dtau / 3.0) ** 0.5 * 0.5
        exp_tau = abs(tau_start - np.sum(exp_tau_grad, axis=0))

        #exp_probs = np.array([[ 0.49521008, 0.50478989], [0.58135545, 0.41864461], [0.5, 0.5]])

        tau_sq_z_diff = np.array(
            [[[-0.5, -0.625], [8.0, -9.0]], [[2.0, 0], [-2.0, -31.5]], [[3.5, 0.375], [20.0, 18.0]]], dtype=np.float32)
        logits = np.array([[-8.125, -5.28125], [-2.5, -55.125], [-56.125, -18.28125]], dtype=np.float32)
        rbf = np.exp(logits)
        exps = np.exp(4.0 * rbf)
        a = exps / np.sum(exps, axis=1).reshape(3, 1)
        z_sm_grad = y_hot * (1 - a).reshape([3, 1, 2])
        mag = np.sum(tau_sq_z_diff ** 2.0, axis=1) ** 0.5
        mag_shaped = mag.reshape([3, 1, 2])
        dz = z_sm_grad * -tau_sq_z_diff / mag_shaped
        exp_z_grad = - 2.0 ** 0.5 * dz

        z_bar_sm_grad = (y_hot - a.reshape([3, 1, 2]))
        exp_z_bar_grad = -z_bar_sm_grad * tau_sq_z_diff / mag_shaped / 3 ** 0.5

        self.assertTrue(np.allclose(exp_z_diff_sq, z_diff_sq))
        self.assertTrue(np.allclose(exp_tau_sq, tau_sq))
        self.assertTrue(np.allclose(exp_wzds, wzds))
        self.assertTrue(np.allclose(exp_wzds, wzdso))
        self.assertTrue(np.allclose(exp_tau_grad, tau_grad))
        self.assertTrue(np.allclose(exp_tau, tau))

        self.assertTrue(np.allclose(exp_z_grad, z_grad))

        self.assertTrue(np.allclose(exp_z_bar_grad, z_bar_grad))


if __name__ == '__main__':
    test.main()