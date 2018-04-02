import tensorflow as tf
import numpy as np
from visualize import *

epochs = 100

class Net:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[2, 2])
        self.w = tf.get_variable("w", shape=[2, 1], initializer=tf.constant_initializer(np.array([1.0, 1.0])))
        self.b = tf.get_variable("b", shape=[1], initializer=tf.constant_initializer([2.5]))
        self.w_soft = tf.get_variable("w_soft", shape=[1, 2], initializer=tf.constant_initializer(np.array([1.0, 1.0])))
        a = tf.matmul(self.x, self.w) + self.b
        a = tf.nn.relu(a)
        a = tf.matmul(a, self.w_soft)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=[[0, 1], [1, 0]], logits=[a])
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

    def params(self):
        return self.w, self.b, self.w_soft

net = Net()
params_op = net.params()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
input = np.array([[-1.0, -1.0], [1.0, 1.0]])

ws = []
bs = []
w_softs = []
for i in xrange(epochs):
    sess.run(net.train_op, feed_dict={net.x: input})
    w, b, w_soft = sess.run(params_op)
    ws.append(w)
    bs.append(b)
    w_softs.append(w_soft)

show_net_over_time(ws, bs, w_softs)