import tensorflow as tf
import rbf as rb

class Network:

    def __init__(self, rbf, conf):
        self.rbf = rbf
        d = conf.d
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

        core_ops = self.rbf.create_ops(self.z).core_ops()
        self.train_op = core_ops[0]
        self.z = core_ops[1]
        self.z_bar = core_ops[2]
        self.tau = core_ops[3]
        self.a = core_ops[4]


        #Compute the loss and apply the optimiser
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=a)
        # self.train_op = conf.optimizer(learning_rate=conf.lr).minimize(loss)
        # #Produce probabilities for accuracy
        # self.a = tf.nn.softmax(a)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.rbf.y

    def get_batch_size(self):
        return rb.batch_size

    def rbf_params(self):
        return self.z, self.z_bar, self.tau

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

