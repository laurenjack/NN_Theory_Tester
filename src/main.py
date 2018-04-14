import tensorflow as tf
import numpy as np
from test_rbf import RBF
from gen_train_points import *
from animator import *

class Conf:
    pass

conf = Conf()
conf.out_dir='/Users/jack/tf_runs/test_rbf4'
conf.n = 100
conf.num_class = 5
conf.d = 2
conf.rbf_c = 10.0
conf.z_bar_init_sd = 3.0
conf.z_sd = 6.0
conf.lr = 0.2
conf.show_details = False
conf.show_animation = True
conf.train_centres_taus = True
conf.optimizer = tf.train.AdamOptimizer
epochs = 500

net = RBF(conf)
all_ops = net.all_ops()

#Summaries for variables
for var in tf.trainable_variables():
    zeros = tf.zeros(shape=var.shape)
    tf.summary.histogram(var.op.name, tf.where(tf.is_nan(var), zeros, var))
summary_op = tf.summary.merge_all()
all_ops.append(summary_op)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
y = gen_train_labels(conf)

summary_writer = tf.summary.FileWriter(conf.out_dir, sess.graph)

class_wise_z_list = []
for k in xrange(conf.num_class):
    class_wise_z_list.append([])
z_bar_list = []
tau_list = []

for e in xrange(epochs):
    _, z, z_bar, tau, a, summ_str = sess.run(all_ops, feed_dict={net.y: y})
    for k in xrange(conf.num_class):
        ind_of_class = np.argwhere(y == k)[:, 0]
        class_wise_z_list[k].append(z[ind_of_class])
    z_bar_list.append(z_bar)
    tau_list.append(tau)
    summary_writer.add_summary(summ_str, e)

#Take the last a and evaluate the percentage of correctly classified points
max_a = np.amax(a, axis=1)
arg_max_a = np.argmax(a, axis=1)
correct_responses = np.where(np.logical_and(y == arg_max_a, max_a > 0.9), 1, 0)
percentage_correct = np.sum(correct_responses) / float(y.shape[0])
print percentage_correct * 100

if conf.d == 2:
    animate(class_wise_z_list, z_bar_list, tau_list, conf)

