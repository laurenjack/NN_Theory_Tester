import tensorflow as tf


class Network(object):
    """Abstract class representing a neural network.
    """

    def __init__(self, end, input_shape, is_resnet, model_save_dir=None):
        """
        Args:
            conf: see configuration.RbfSoftmaxConfiguration
            end: The last layer/part of the network, e.g. a softmax end with a cross entropy loss function.
            input_shape: The batch-wise input shape for the networks inputs e.g. [None, 32, 32, 3] (CIFAR10)
        """
        self.end = end
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        self.x = tf.placeholder(tf.float32, shape=input_shape, name="inputs")
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')
        self.is_resnet = is_resnet
        self.model_save_dir = model_save_dir

    def gradient_wrt_inputs(self):
        """Find the gradient of the loss function with respect to network inputs.

        This method is useful for generating adversarial examples.

        Returns: A tensor with the same shape as the inputs, d_loss / d_x.
        """
        return tf.gradients(self.loss, self.x)[0]

    # TODO(Jack) refactor this bad boy out for good
    def rbf_params(self):
        if not self.has_rbf():
            raise NotImplementedError('This network does not have an rbf end')
        return self.all_end_ops[2], self.all_end_ops[3], self.all_end_ops[4]