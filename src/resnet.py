import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

BATCH_NORM_OPS_KEY = 'batch_norm_ops'


class Resnet:

    def __init__(self, conf):
        self.kernel_stride = conf.kernel_stride
        self.stack_entry_kernel_stride = conf.stack_entry_kernel_stride
        self.kernel_width = conf.kernel_width
        self.bn_decay = conf.bn_decay
        self.bn_epsilon = conf.bn_epsilon
        self.weight_decay = conf.weight_decay
        self.num_class = conf.num_class

        self.x = tf.placeholder(tf.float32, shape=[None, conf.image_width, conf.image_width, conf.image_depth],
                                name="x")
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

        # Create the first layer
        a = self._layer(self.x, conf.num_filter_first, self.kernel_stride, 'first_layer')
        # Create other hidden layers
        num_stack = len(conf.num_filter_for_stack)
        for num_filter, num_block, i in zip(conf.num_filter_for_stack, conf.num_blocks_per_stack, xrange(num_stack)):
            a = self._stack(a, num_filter, num_block, i)

        a = tf.reduce_mean(a, axis=[1, 2], name="avg_pool")
        a = self._fc(a)
        self.a = tf.nn.softmax(a)
        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=a)
        self.main_loss = tf.reduce_mean(xe)

        # Regularisation (still tied to the lr of the main update)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = self.weight_decay * tf.add_n(reg_losses)

        # Batch Normalisation ops
        batch_norm_updates = tf.get_collection(BATCH_NORM_OPS_KEY)
        batch_norm_updates_op = tf.group(*batch_norm_updates)

        self.loss = self.main_loss + reg_loss
        self.optimzer = conf.optimizer(learning_rate=self.lr).minimize(self.loss)
        self.train_op = tf.group(self.optimzer, batch_norm_updates_op)

    def has_rbf(self):
        return False

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_lr(self):
        return self.lr

    def _stack(self, a, num_filter, num_block, stack_index):
        scope_name = 'stack'+str(stack_index)
        with tf.variable_scope(scope_name):
            # transition to this stack, by creating the first block
            if scope_name == 'stack0':
                a = self._block(a, num_filter, "block0")
            else:
                with tf.variable_scope("block0"):
                    skip = tf.layers.average_pooling2d(a, 2, 2)  # self._layer(a, stack.in_d, 1, 2, "skip_projection")
                    prev_out = tf.shape(a)[3]
                    pad_amount = (num_filter - prev_out) // 2
                    skip = tf.pad(skip, [[0, 0], [0, 0], [0, 0], [pad_amount, pad_amount]])
                    a = self._layer(a, num_filter, self.stack_entry_kernel_stride, "layer_0")
                    a = self._layer(a, num_filter, self.kernel_stride, "layer_1", skip=skip)
            for b in xrange(num_block-1):
                block_scope = "block"+str(b+1)
                a = self._block(a, num_filter, block_scope)
            return a

    def _block(self, a, num_filter, scope_name):
        with tf.variable_scope(scope_name):
            skip = a
            a = self._layer(a, num_filter, self.kernel_stride, "layer_0")
            a = self._layer(a, num_filter, self.kernel_stride, "layer_1", skip=skip)
            return a

    def _layer(self, a, num_filter, stride, scope_name, skip=None):
        with tf.variable_scope(scope_name):
            a = self._convolution(a, num_filter, stride)
            a = self._bn(a)
            if skip is not None:
                a = a + skip
        return tf.nn.relu(a)

    def _convolution(self, a, filters_out, stride):
        filters_in = a.get_shape()[-1]
        shape = [self.kernel_width, self.kernel_width, filters_in, filters_out]
        initializer = tf.contrib.layers.variance_scaling_initializer(mode='FAN_OUT')
        weights = self._get_variable('weights', shape=shape, initializer=initializer)
        weight_reg = tf.nn.l2_loss(weights)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_reg)
        return tf.nn.conv2d(a, weights, [1, stride, stride, 1], padding='SAME')

    def _bn(self, a):
        a_shape = a.get_shape()
        params_shape = a_shape[-1:]
        axis = list(range(len(a_shape) - 1))

        beta = self._get_variable('beta', params_shape, initializer=tf.zeros_initializer)
        gamma = self._get_variable('gamma', params_shape, initializer=tf.ones_initializer)
        moving_mean = self._get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
        moving_variance = self._get_variable('moving_variance', params_shape, initializer=tf.ones_initializer,
                                        trainable=False)

        # These ops will only be performed when training.
        mean, variance = tf.nn.moments(a, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, self.bn_decay)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, self.bn_decay)
        tf.add_to_collection(BATCH_NORM_OPS_KEY, update_moving_mean)
        tf.add_to_collection(BATCH_NORM_OPS_KEY, update_moving_variance)

        mean, variance = control_flow_ops.cond(
            self.is_training, lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

        a = tf.nn.batch_normalization(a, mean, variance, beta, gamma, self.bn_epsilon)
        # x.set_shape(inputs.get_shape()) ??

        return a

    def _fc(self, a):
        num_units_in = a.get_shape()[1]
        num_units_out = self.num_class
        weights_initializer = tf.contrib.layers.variance_scaling_initializer()

        weights = self._get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer)
        biases = self._get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer)
        a = tf.nn.xw_plus_b(a, weights, biases)
        weight_reg = tf.nn.l2_loss(weights)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_reg)
        return a

    def _get_variable(self, name, shape, initializer, dtype=tf.float32, trainable=True):
        """A little wrapper around tf.get_variable to do weight decay and add to resnet collection"""
        collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype,
                               collections=collections, trainable=trainable)
