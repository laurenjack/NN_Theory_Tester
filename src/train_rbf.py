import tensorflow as tf
import numpy as np
from rbf import RBF
from gen_train_points import *

def train(conf):
    net = RBF(conf)
    all_ops = net.all_ops()

    # Summaries for variables
    if conf.num_runs == 1:
        for var in tf.trainable_variables():
            zeros = tf.zeros(shape=var.shape)
            tf.summary.histogram(var.op.name, tf.where(tf.is_nan(var), var, var))
        summary_op = tf.summary.merge_all()
        all_ops.append(summary_op)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    y = gen_train_labels(conf)

    if conf.num_runs == 1:
        summary_writer = tf.summary.FileWriter(conf.out_dir, sess.graph)

    class_wise_z_list = None
    z_bar_list = None
    tau_list = None
    if conf.num_runs == 1:
        class_wise_z_list = []
        for k in xrange(conf.num_class):
            class_wise_z_list.append([])
        z_bar_list = []
        tau_list = []

    for e in xrange(conf.epochs):

        if conf.num_runs == 1:
            _, z, z_bar, tau, a, summ_str = sess.run(all_ops, feed_dict={net.y: y})
            for k in xrange(conf.num_class):
                ind_of_class = np.argwhere(y == k)[:, 0]
                class_wise_z_list[k].append(z[ind_of_class])
            z_bar_list.append(z_bar)
            tau_list.append(tau)
            summary_writer.add_summary(summ_str, e)
        else:
            _, z, z_bar, tau, a = sess.run(all_ops, feed_dict={net.y: y})



    max_a = np.amax(a, axis=1)
    arg_max_a = np.argmax(a, axis=1)
    correct_responses = np.where(np.logical_and(y == arg_max_a, max_a > conf.classified_as_thresh), 1, 0)
    num_correct = np.sum(correct_responses)

    sess.close()
    tf.reset_default_graph()

    return TrainResult(num_correct, class_wise_z_list, z_bar_list, tau_list)

class TrainResult:

    def __init__(self, num_correct, class_wise_z_list=None, z_bar_list=None, tau_list=None):
        self.num_correct = num_correct
        self.class_wise_z_list = class_wise_z_list
        self.z_bar_list = z_bar_list
        self.tau_list = tau_list

    def get(self):
        return self.class_wise_z_list, self.z_bar_list, self.tau_list
