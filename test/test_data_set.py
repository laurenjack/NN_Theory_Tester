import unittest

import src.data_set as ds
from src.rbf_softmax import configuration


class TestDataSet(unittest.TestCase):

    def test_when_load_two_classes_cifar10(self):
        # Not ideal, but using this conf singleton for the data directory (should mock loading of data)
        conf = configuration.get_configuration()
        data_dir = conf.data_dir

        classes = (0, 7)
        data_set = ds.load_cifar(data_dir, classes)

        self.assertEqual(src.data_set.train.n, 9000)
        self.assertEqual(src.data_set.validation.n, 1000)
        self._assert_all_examples_are_of_class(src.data_set.train.y, classes)
        self._assert_all_examples_are_of_class(src.data_set.validation.y, classes)

    def _assert_all_examples_are_of_class(self, y, classes):
        for i in xrange(y.shape[0]):
            self.assertTrue(y[i] == classes[0] or y[i] == classes[1])
