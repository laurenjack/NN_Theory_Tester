import tensorflow as tf
import numpy as np
import unittest as test
from mock import *
from bar_network import *

class TestBarNetwork(test.TestCase):

    def _get_variable_side_effect(self, name, shape=None, initializer=None):
        if name.startswith('b'):
            return tf.constant(1, dtype=tf.float32, shape=shape, name=name)
        return tf.constant(next(self.weight_sequence), dtype=tf.float32, shape=shape, name=name)

    def test_feedforward(self):
        x = np.array([[[-0.5, 1],
                       [2, 3]]])

        # Mock up weights which would otherwise be created by a tf initialiser
        w1 = list(np.array([[2, -0.1], [0.5, 4], [2, 1]]))
        w2 = list(np.array([[3.0, 5, -0.4], [0.6, -0.3, 0.7]]))
        self.weight_sequence = iter(w1 + w2)
        vc_mock = Mock()
        vc_mock.get_variable = Mock()
        vc_mock.get_variable.side_effect = self._get_variable_side_effect

        # Specify the expected output of the second conv layer
        a2_expected = np.array([[43.95, 74.2], [0.0, 2.77]])
        bn = BarNetwork(vc_mock, [3, 2])
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        a2_result = sess.run(bn.a2, feed_dict={bn.x: x, bn.y: [0]})
        print a2_result

        self.assertTrue(np.allclose(a2_expected, a2_result))


