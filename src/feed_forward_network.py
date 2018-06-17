import tensorflow as tf

class Network:

    def __init__(self, conf):
        #Set up placeholders for inputs and putputs
        num_inputs = conf.num_inputs
        num_class = conf.num_class
        self.x = tf.placeholder(tf.float32, shape=[None, num_inputs], name="inputs")
        self.y = tf.placeholder(tf.int32, shape=[None], name="target_outputs")

        #Create the feedforward component
        hidden_sizes = conf.hidden_sizes
        ins = [num_inputs] + hidden_sizes
        outs = hidden_sizes + [num_class]
        a = self.x
        for l, inp, out in zip(range(len(outs)), ins, outs):
            a = self._create_layer(a, l, [inp, out])

        self.logits = a



        #Compute the loss and apply the optimiser
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=a)
        self.train_op = conf.optimizer(learning_rate=conf.lr).minimize(loss)
        #Produce probabilities for accuracy
        self.a = tf.nn.softmax(a)

    def _create_layer(self, a, l, shape):
        weights_init = tf.contrib.layers.variance_scaling_initializer()
        W = tf.get_variable('W'+str(l),
                        shape=shape,
                        initializer=weights_init)
        bias_init = tf.zeros_initializer()
        b = tf.get_variable('b'+str(l),
                            shape[1],
                            initializer=bias_init)
        a = tf.nn.xw_plus_b(a, W, b)
        return tf.nn.relu(a)

