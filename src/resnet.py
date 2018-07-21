import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

BATCH_NORM_OPS_KEY = 'batch_norm_ops'


class Resnet:

    def __init__(self, conf, end):
        self.end = end
        self.kernel_stride = conf.kernel_stride
        self.stack_entry_kernel_stride = conf.stack_entry_kernel_stride
        self.kernel_width = conf.kernel_width
        self.bn_decay = conf.bn_decay
        self.bn_epsilon = conf.bn_epsilon
        self.num_class = conf.num_class
        self.image_width = conf.image_width

        self.x = tf.placeholder(tf.float32, shape=[None, conf.image_width, conf.image_width, conf.image_depth],
                                name="x")
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')
        self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
        epochs_trained = tf.Variable(0, trainable=False, name='epochs_trained', dtype=tf.int32)
        increment_epochs_trained = tf.assign(epochs_trained, epochs_trained+1)
        self.weight_decay = tf.train.exponential_decay(conf.weight_decay, epochs_trained,
                                                       conf.decay_epochs, conf.wd_decay_rate)

        # Create the first layer
        x_augmented = self._augment(self.x)
        a = self._layer(x_augmented, conf.num_filter_first, self.kernel_stride, 'first_layer')
        # Create other hidden layers
        num_stack = len(conf.num_filter_for_stack)
        for num_filter, num_block, i in zip(conf.num_filter_for_stack, conf.num_blocks_per_stack, xrange(num_stack)):
            a = self._stack(a, num_filter, num_block, i)

        a = tf.reduce_mean(a, axis=[1, 2], name="avg_pool")
        self.a, main_loss = end.create_ops(a)

        # Regularisation (still tied to the lr of the main update)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = self.weight_decay * tf.add_n(reg_losses)

        # Batch Normalisation ops
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # batch_norm_updates = tf.get_collection(BATCH_NORM_OPS_KEY)
        # batch_norm_updates_op = tf.group(*batch_norm_updates)
        self.loss = main_loss + reg_loss
        self.optimzer = conf.optimizer(learning_rate=self.lr, momentum=0.9).minimize(self.loss)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.group(self.optimzer, increment_epochs_trained)

    def has_rbf(self):
        return False

    def get_x(self):
        return self.x

    def get_y(self):
        return self.end.y

    def get_lr(self):
        return self.lr

    def get_batch_size(self):
        return self.batch_size

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
            if skip is None:
                a = tf.layers.batch_normalization(a, momentum=0.9, epsilon=1e-5, training=self.is_training)
                a = tf.nn.relu(a)
            else:
                a = a + skip
            return a

    def _convolution(self, a, filters_out, stride):
        filters_in = a.get_shape()[-1]
        shape = [self.kernel_width, self.kernel_width, filters_in, filters_out]
        initializer = tf.contrib.layers.variance_scaling_initializer(2.0)
        weights = self._get_variable('weights', shape=shape, initializer=initializer)
        weight_reg = tf.nn.l2_loss(weights)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_reg)
        return tf.nn.conv2d(a, weights, [1, stride, stride, 1], padding='SAME')

    def _augment(self, x):
        padded = tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]])
        cropped = tf.random_crop(padded, [self.batch_size, self.image_width, self.image_width, 3])
        do_flip = tf.greater(tf.random_uniform(shape=[self.batch_size]), 0.5)
        flipped = tf.where(do_flip, tf.image.flip_left_right(cropped), cropped)
        return control_flow_ops.cond(self.is_training, lambda: flipped, lambda: x)


    def _get_variable(self, name, shape, initializer, dtype=tf.float32, trainable=True):
        """A little wrapper around tf.get_variable to do weight decay and add to resnet collection"""
        collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype,
                               collections=collections, trainable=trainable)
