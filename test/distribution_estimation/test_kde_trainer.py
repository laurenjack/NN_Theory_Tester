import unittest
import mock
import numpy as np

from test import test_utils
from src.distribution_estimation import kde_trainer


class TestKdeTrainer(unittest.TestCase):

    def test_when_samples_not_multiples_of_n_then_trains_to_the_floor(self):
        # Create Mocks
        kde = mock.Mock()
        kde.squared_weighted_mean_error = mock.Mock(return_value=('train_op', 'cost', 'h_tensor', 'gradient'))
        kde.a = 'a'
        kde.a_star = 'a_star'

        conf = mock.Mock()
        conf.epochs = 2
        conf.n = 12
        conf.m = 3
        conf.r = 2

        sess = mock.Mock()
        sess.run = mock.Mock(return_value=(None, 0, 'Test_h', 'Test_gradient'))

        random = mock.Mock()
        fake_distribution = 3 * np.arange(12)
        random.normal_numpy_array = mock.Mock(return_value=fake_distribution)
        # Shuffle doesn't need to do anything
        random.shuffle = mock.Mock()

        # Run the Kde trainer
        kde_trainer.train(kde, conf, sess, random)

        # Assert that session.run was called with the correct elements
        random.normal_numpy_array.assert_called_once_with([12])
        expected_tensor_args = [['train_op', 'cost', 'h_tensor', 'gradient']] * 4
        expected_feed_dicts = [{'a': np.array([0, 3, 6]), 'a_star': np.array([9, 12])},
                               {'a': np.array([15, 18, 21]), 'a_star': np.array([24, 27])},
                               {'a': np.array([0, 3, 6]), 'a_star': np.array([9, 12])},
                               {'a': np.array([15, 18, 21]), 'a_star': np.array([24, 27])}]
        test_utils.assert_sess_run_called_with(sess, expected_tensor_args, expected_feed_dicts)