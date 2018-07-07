import tensorflow as tf
import rbf as rb

class Network:

    def __init__(self, rbf, conf):
        self.rbf = rbf
        d = conf.d
        self.adverserial_epsilon = conf.adverserial_epsilon
        #Set up placeholders for inputs and putputs
        num_inputs = conf.num_inputs
        num_class = conf.num_class
        self.x = tf.placeholder(tf.float32, shape=[None, num_inputs], name="inputs")
        # self.y = tf.placeholder(tf.int32, shape=[None], name="target_outputs")

        #Create the feedforward component
        hidden_sizes = conf.hidden_sizes
        ins = [num_inputs] + hidden_sizes
        outs = hidden_sizes + [d]
        a = self.x
        for l, inp, out in zip(range(len(outs[:-1])), ins[:-1], outs[:-1]):
            a = self._create_layer(a, l, [inp, out], activation_func=tf.nn.relu)

        self.z = self._create_layer(a, l+1, [ins[-1], outs[-1]])

        rbf_ops = self.rbf.create_ops(self.z)
        core_ops = rbf_ops.core_ops()
        self.train_op = core_ops[0]
        self.z = core_ops[1]
        self.z_bar = core_ops[2]
        self.tau = core_ops[3]
        self.a = core_ops[4]
        self.loss = rbf_ops.loss

        #Compute the loss and apply the optimiser
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=a)
        # self.train_op = conf.optimizer(learning_rate=conf.lr).minimize(loss)
        # #Produce probabilities for accuracy
        # self.a = tf.nn.softmax(a)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.rbf.y

    def get_y_hot(self):
        return self.rbf.y_hot

    def get_batch_size(self):
        return self.rbf.batch_size

    def get_lr(self):
        return self.rbf.lr

    def rbf_params(self):
        return self.z, self.z_bar, self.tau

    def fgsm_adverserial_with_target(self):
        """Generate an adverserial example using the fast gradient sign method.
        See doc for adverserial_gradient for more info """
        image_grad = self.adverserial_gradient()
        grad_sign = tf.sign(image_grad)
        pertubation = tf.multiply(self.adverserial_epsilon, grad_sign)
        return self.x - pertubation

    def adverserial_gradient(self):
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

