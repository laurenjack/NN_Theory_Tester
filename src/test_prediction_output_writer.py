import unittest as test
from mock import *
from network_runner import PredictionReport
import numpy as np
from prediction_analytics import *

class TestPredictionOutputWriter(test.TestCase):

    def test_extract_transform(self):
        #Test data
        a_correct = np.array([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.15, 0.15, 0.7]])
        y_correct = np.array([1, 0, 2])
        correct = PredictionReport("Correct", a_correct, None, y_correct)
        a_incorrect = np.array([[0.16, 0.34, 0.5], [0.2, 0.6, 0.2]])
        y_incorrect = np.array([0, 2])
        incorrect = PredictionReport("Incorrect", a_incorrect, None, y_incorrect)

        z = np.array([[1, 2, 3, 4, 5],
                      [6, 7, 8, 9, 10]], np.float32).transpose()
        z_bar = np.array([[11, 12], [13, 14], [15, 16]], np.float32).transpose()
        tau = np.array([[111, 222], [333, 444], [555, 666]], np.float32).transpose()

        #Mocks
        corr_inds = np.array([99, 88, 77])
        incorr_inds = np.array([66, 55])
        all_inds = np.array([99, 88, 77, 66, 55])
        X = Mock()
        Y = Mock()
        X.shape.__getitem__ = Mock(return_value=5)
        X.__getitem__ = Mock(return_value='mock X')
        Y.__getitem__ = Mock(return_value='mock Y')
        network_runner = Mock()
        network_runner.all_correct_incorrect = Mock(return_value=(correct, incorrect, corr_inds, incorr_inds))
        network_runner.report_rbf_params = Mock(return_value=((z, z_bar, tau)))

        #Run the code
        point_stat, dimension_stat = extract_and_transform(X, Y, network_runner)

        #mock assertions
        X.shape.__getitem__.assert_called_once_with(0)
        network_runner.all_correct_incorrect.assert_called_once_with(X, Y)
        network_runner.report_rbf_params.assert_called_once_with('mock X', 'mock Y')

        #Data assertions
        exp_point_stat = np.array([[0, 1, 2, 3, 4],
                                  [1, 0, 2, 2, 1],
                                  [0.8, 0.8, 0.7, 0.5, 0.6],
                                  [True, True, True, False, False]], dtype=object)

        exp_dimension_stat = np.array([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                                       [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                      [-12, -8, -9, -5, -12, -8, -11, -7, -8, -4],
                                      [333, 444, 111, 222, 555, 666, 555, 666, 333, 444],
                                       [1, 6, 2, 7, 3, 8, 4, 9, 5, 10],
                                       [13, 14, 11, 12, 15, 16, 15, 16, 13, 14]], dtype=object)

        self.assertTrue(np.array_equal(exp_point_stat, point_stat))
        self.assertTrue(np.array_equal(exp_dimension_stat, dimension_stat))




