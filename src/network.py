import tensorflow as tf


class Network(object):
    """Abstract class representing a neural network.
    """

    def __init__(self, end, input_shape, is_resnet, model_save_dir):
        """
        Args:
            end: The last layer/part of the network, e.g. a softmax end with a cross entropy loss function.
            input_shape: The batch-wise shape for the networks inputs e.g. [None, 32, 32, 3] (CIFAR10)
            is_resnet: True if the network is a resnet
            model_save_dir: The directory to save all the network's variables.
        """
        self.end = end
        self.activation_list = []
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        self.x = tf.placeholder(tf.float32, shape=input_shape, name="inputs")
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')
        self.is_resnet = is_resnet
        self.model_save_dir = model_save_dir

    def get_tensors(self):
        """Returns: All the tensors required for training and reporting on this network.
        """
        return [self.train_op] + self.all_end_tensors

    def gradient_wrt_inputs(self):
        """Find the gradient of the loss function with respect to network inputs.

        This method is useful for generating adversarial examples.

        Returns: A tensor with the same shape as the inputs, d_loss / d_x.
        """
        return tf.gradients(self.loss, self.x)[0]

    @property
    def y(self):
        """Placeholder for target classes, shape: [m]."""
        return self.end.y

    @property
    def batch_size(self):
        """Placeholder for the batch size"""
        return self.end.batch_size

    # TODO(Jack) refactor this bad boy out for good
    def has_rbf(self):
        from src.rbf_softmax import rbf
        return isinstance(self.end, rbf.Rbf)

    # TODO(Jack) refactor this bad boy out for good
    def rbf_params(self):
        """rbf_tensors: A tuple of three tensorflow tensors, z [m, d], z_bar [d, num_class] and tau [d, num_class]"""
        if not self.has_rbf():
            raise NotImplementedError('This network does not have an rbf end')
        return self.all_end_tensors[2], self.all_end_tensors[3], self.all_end_tensors[4]