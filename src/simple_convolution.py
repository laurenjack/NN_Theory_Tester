import tensorflow as tf
import operation


_KERNEL_WIDTH = 3


class SimpleConvolutionService(object):

    def __init__(self, image_width):
        self.weight_decay = 0.0001
        self.image_width = image_width
        self.full_vector_size = image_width * image_width

    def get_tensors(self):
        activations = []
        lr = tf.placeholder(tf.float32, shape=[], name="lr")
        x = tf.placeholder(tf.float32, shape=[None, self.full_vector_size], name='x')
        y = tf.placeholder(tf.int64, shape=[None], name="y")
        x_image = tf.reshape(x, shape=[-1, self.image_width, self.image_width, 1])
        activations.append(x_image)
        a = self._convolution(x_image, 32, 2, 'first', kernel_width=5)
        a = tf.layers.max_pooling2d(a, 2, 2)
        activations.append(a)
        a = self._convolution(a, 64, 2, 'second')
        a = tf.layers.max_pooling2d(a, 2, 2)
        activations.append(a)
        a = tf.reduce_mean(a, axis=[1, 2], name='avg_pool')
        activations.append(a)
        # a = tf.contrib.layers.flatten(a)
        a = operation.fc(a, 64)
        activations.append(a)
        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=a)
        prediction = tf.argmax(tf.nn.softmax(logits=a), axis=1)
        correct_indicator = tf.cast(tf.equal(y, prediction), tf.float32)
        accuracy = tf.reduce_mean(correct_indicator)
        main_loss = tf.reduce_mean(xe)
        # Regularisation (still tied to the lr of the main update)
        # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # reg_loss = self.weight_decay * tf.add_n(reg_losses)
        loss = main_loss # + reg_loss
        train = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss)
        return lr, x, y, train, activations, accuracy



    def _convolution(self, a, filters_out, stride, name, kernel_width=_KERNEL_WIDTH):
        filters_in = a.get_shape()[-1]
        shape = [kernel_width, kernel_width, filters_in, filters_out]
        initializer = tf.contrib.layers.variance_scaling_initializer(2.0)
        weights = tf.get_variable(name, shape=shape, initializer=initializer)
        weight_reg = tf.nn.l2_loss(weights)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_reg)
        return tf.nn.conv2d(a, weights, [1, stride, stride, 1], padding='SAME')




class SimpleConvolutionGraph(object):

    def __init__(self):
        pass
