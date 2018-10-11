import tensorflow as tf
from rbf import RBF
import configuration
conf = configuration.get_configuration()

class Network:

    def __init__(self, num_inputs, end):
        self.model_save_dir = None
        self.end = end
        d = conf.d
        self.adversarial_epsilon = conf.adversarial_epsilon
        #Set up placeholders for inputs and putputs
        num_class = conf.num_class
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        self.x = tf.placeholder(tf.float32, shape=[None, num_inputs], name="inputs")
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

        #Create the feedforward component
        hidden_sizes = conf.hidden_sizes
        ins = [num_inputs] + hidden_sizes
        outs = hidden_sizes + [d]
        a = self.x
        for l, inp, out in zip(range(len(outs[:-1])), ins[:-1], outs[:-1]):
            a = self._create_layer(a, l, [inp, out], activation_func=tf.nn.relu)

        self.z = self._create_layer(a, l+1, [ins[-1], outs[-1]])

        self.all_end_ops = self.end.tensors_for_network(self.z)
        self.a = self.all_end_ops[0]
        self.loss = self.all_end_ops[1]
        self.train_op = conf.optimizer(learning_rate=conf.lr, momentum=0.9).minimize(self.loss)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.end.y

    def get_y_hot(self):
        return self.end.y_hot

    def get_batch_size(self):
        return self.end.batch_size

    def get_lr(self):
        return self.lr

    def get_ops(self):
        return [self.train_op] + self.all_end_ops

    def rbf_params(self):
        if not self.has_rbf():
            raise NotImplementedError('This network does not have an rbf end')
        return self.all_end_ops[2], self.all_end_ops[3], self.all_end_ops[4]

    def has_rbf(self):
        return isinstance(self.end, RBF)

    def fgsm_adversarial_with_target(self):
        """Generate an adversarial example using the fast gradient sign method.
        See doc for adversarial_gradient for more info """
        image_grad = self.adversarial_gradient()
        grad_sign = tf.sign(image_grad)
        pertubation = tf.multiply(self.adversarial_epsilon, grad_sign)
        return self.x - pertubation

    def adversarial_gradient(self):
        """ Given an image and a deliberately faulty label, return the gradient
        with respect to that image which minimises the loss function, using that
        faulty label.

        As the z grads ignore off target points, simply trying to
        maximise the loss function will just run the example down into a low
        rbf region. Therefore, rather than maximising the loss, we minimise
        the loss with respect to a different target. Ideally the nearest z_bar"""
        return tf.gradients(self.loss, self.x)[0]

    def _create_layer(self, a, l, shape, activation_func=None):
        weights_init = tf.contrib.layers.variance_scaling_initializer()
        W = tf.get_variable('W'+str(l),
                        shape=shape,
                        initializer=weights_init)
        bias_init = tf.zeros_initializer()
        b = tf.get_variable('b'+str(l),
                            shape[1],
                            initializer=bias_init)
        a = tf.nn.xw_plus_b(a, W, b)
        if activation_func is not None:
            return activation_func(a)
        return a

