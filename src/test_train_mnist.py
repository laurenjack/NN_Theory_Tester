import unittest as test
from mock import *
import sys
mock_mnist = mock.Mock()
sys.modules['tensorflow.examples.tutorials.mnist'] = mock_mnist
mock_mnist.input_data = mock.Mock()
from feed_forward_network import FeedForward
from train_network import *

class TestTrainMnist(test.TestCase):

    def test_accuracy(self):
        conf = mock.Mock()
        network = mock.Mock()
        network.train = 'train_mock'
        network.x = 'x_mock'
        network.y = 'y_mock'

        class MockData:

            def __init__(self, images, labels):
                self.images = images
                self.labels = labels

        mock_ds = Mock()
        mock_ds.train = MockData(np.array([5,6,7,8,9]), np.array([1, 1, 1, 0, 0]))
        mock_ds.validation = MockData(np.array([9, 8, 7, 6]), np.array([0, 0, 1, 1]))
        mock_mnist.input_data.read_data_sets = MagicMock(return_value=mock_ds)



        train_accs, val_accs = train(network, conf)
        print "Done"


if __name__ == '__main__':
    test.main()

