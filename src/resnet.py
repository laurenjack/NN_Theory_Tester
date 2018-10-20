import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import network
import operation


_IMAGE_DEPTH = 3
_KERNEL_WIDTH = 3
_KERNEL_STRIDE = 1
_STACK_ENTRY_KERNEL_STRIDE = 2  # The kernel stride when moving from one stack to the next thinner stack.


class Resnet(network.Network):
    """Represents a Residual Neural network, the architecture is similar to: https://arxiv.org/abs/1512.03385
    """

    def __init__(self, conf, end, model_save_dir, image_width):
        """
        Args:
            conf: A static set of properties to configure the network
            end: The last layer/part of the network, e.g. an rbf-softmax end with a cross entropy loss function
            model_save_dir: The directory to save all this network's variables.
            image_width: The width (and assumed identical height) of the image input.
        """
        input_shape = [None, image_width, image_width, _IMAGE_DEPTH]
        super(Resnet, self).__init__(end, input_shape, True, model_save_dir)

        self.bn_decay = conf.bn_decay
        self.bn_epsilon = conf.bn_epsilon

        training_steps = tf.Variable(0, trainable=False, name='training_steps', dtype=tf.int32)
        increment_epochs_trained = tf.assign(training_steps, training_steps+1)
        self.weight_decay = tf.train.exponential_decay(conf.weight_decay, training_steps,
                                                       conf.decay_epochs, conf.wd_decay_rate)

        # Create the first layer
        x_augmented = self._augment(self.x, image_width)
        a = self._layer(x_augmented, conf.num_filter_first, _KERNEL_STRIDE, 'first_layer')

        # Create other hidden layers
        num_stack = len(conf.num_filter_for_stack)
        for num_filter, num_block, i in zip(conf.num_filter_for_stack, conf.num_blocks_per_stack, xrange(num_stack)):
            a = self._stack(a, num_filter, num_block, i)

        if conf.is_per_filter_fc:
            z = operation.per_filter_fc(a, conf.d)
        else:
            a = tf.reduce_mean(a, axis=[1, 2], name="avg_pool")
            z = operation.fc(a, conf.d)

        self.all_end_ops = end.tensors_for_network(z)
        self.a = self.all_end_ops[0]
        main_loss = self.all_end_ops[1]

        # Regularisation (still tied to the lr of the main update)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = self.weight_decay * tf.add_n(reg_losses)

        # Batch Normalisation ops
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.loss = main_loss + reg_loss
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9).minimize(self.loss)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.group(self.optimizer, increment_epochs_trained)

    def _stack(self, a, num_filter, num_block, stack_index):
        """Create a stack and return its outputs.

        A stack is a composed sequence of blocks with the same filter widths and number of filters.
        """
        scope_name = 'stack{}'.format(stack_index)
        with tf.variable_scope(scope_name):
            # transition to this stack, by creating the first block
            if scope_name == 'stack0':
                a = self._block(a, num_filter, "block0")
            else:
                # The first block is in a stack requires padding because the width of filters typically changes
                # between stacks
                with tf.variable_scope("block0"):
                    skip = tf.layers.average_pooling2d(a, 2, 2)
                    prev_out = tf.shape(a)[3]
                    pad_amount = (num_filter - prev_out) // 2
                    skip = tf.pad(skip, [[0, 0], [0, 0], [0, 0], [pad_amount, pad_amount]])
                    a = self._layer(a, num_filter, _STACK_ENTRY_KERNEL_STRIDE, "layer_0")
                    a = self._layer(a, num_filter, _KERNEL_STRIDE, "layer_1", skip=skip)
            for b in xrange(num_block-1):
                block_scope = "block{}".format(b+1)
                a = self._block(a, num_filter, block_scope)
            return a

    def _block(self, a, num_filter, scope_name):
        """Create a block, a block is 2 layers with a skip connection that skips both.
        """
        with tf.variable_scope(scope_name):
            skip = a
            a = self._layer(a, num_filter, _KERNEL_STRIDE, "layer_0")
            a = self._layer(a, num_filter, _KERNEL_STRIDE, "layer_1", skip=skip)
            return a

    def _layer(self, a, num_filter, stride, scope_name, skip=None):
        """Create a single convolutional layer. This applies a convolution, batch normalisations and a ReLU to each
        output.
        """
        with tf.variable_scope(scope_name):
            a = self._convolution(a, num_filter, stride)
            if skip is None:
                a = tf.layers.batch_normalization(a, momentum=0.9, epsilon=1e-5, training=self.is_training)
                a = tf.nn.relu(a)
            else:
                a = a + skip
            return a

    def _convolution(self, a, filters_out, stride):
        filters_in = a.get_shape()[-1]
        shape = [_KERNEL_WIDTH, _KERNEL_WIDTH, filters_in, filters_out]
        initializer = tf.contrib.layers.variance_scaling_initializer(2.0)
        weights = tf.get_variable('weights', shape=shape, initializer=initializer)
        weight_reg = tf.nn.l2_loss(weights)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_reg)
        return tf.nn.conv2d(a, weights, [1, stride, stride, 1], padding='SAME')

    def _augment(self, x, image_width):
        """Make random augmentations to the training data before feeding it to the network"""
        padded = tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]])
        cropped = tf.random_crop(padded, [self.end.batch_size, image_width, image_width, _IMAGE_DEPTH])
        do_flip = tf.greater(tf.random_uniform(shape=[self.end.batch_size]), 0.5)
        flipped = tf.where(do_flip, tf.image.flip_left_right(cropped), cropped)
        return control_flow_ops.cond(self.is_training, lambda: flipped, lambda: x)
