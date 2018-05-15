import tensorflow as tf
import numpy as np
from rbf import RBF
from rbf import num_duds
from gen_train_points import *

def train(conf):
    g_1 = tf.Graph()
    with g_1.as_default():
        net = RBF(conf)
        all_ops = net.all_ops()
        z_bar_op, tau_op = net.z_bar_tau_ops()

        # Summaries for variables
        if conf.num_runs == 1:
            for var in tf.trainable_variables():
                zeros = tf.zeros(shape=var.shape)
                tf.summary.histogram(var.op.name, tf.where(tf.is_nan(var), zeros, var))
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

            z_bar, tau = sess.run([z_bar_op, tau_op])
            all_labels = np.array(reduce(lambda a, b: a + b, [[j] * 20 for j in xrange(10)])).astype(np.int32)
            indicies = np.arange(20 * conf.num_class)
            chosen_indicies = np.random.choice(indicies, size=conf.n, replace=False)
            standard_normals = np.random.randn(20, conf.d, conf.num_class)
            tau = abs(tau.reshape(1, conf.d, conf.num_class))
            z_bar.reshape(1, conf.d, conf.num_class)
            gen_zs_classwise = z_bar + 1.0 / tau * standard_normals
            gen_zs_trans = gen_zs_classwise.transpose(2, 0, 1)
            gen_zs_flattened = gen_zs_trans.reshape(20 * conf.num_class, conf.d)
            gen_zs = gen_zs_flattened[chosen_indicies]
            gen_labels = all_labels[chosen_indicies]
            if e > conf.start_reg:
                ind = 1.0
            else:
                gen_zs = np.ones((conf.n, conf.d))
                ind = 0.0

            if conf.num_runs == 1:
                _, _, z, z_bar, tau, a, summ_str = sess.run(all_ops, feed_dict={net.y: y, net.gen_zs: gen_zs, net.gen_y: gen_labels, net.ind: ind})
                for k in xrange(conf.num_class):
                    ind_of_class = np.argwhere(y == k)[:, 0]
                    ind_of_class_gen = np.argwhere(gen_labels == k)[:, 0]
                    class_zs = np.concatenate([z[ind_of_class], gen_zs[ind_of_class_gen]])
                    class_wise_z_list[k].append(class_zs)
                z_bar_list.append(z_bar)
                tau_list.append(tau)
                summary_writer.add_summary(summ_str, e)
            else:
                _, _, z, z_bar, tau, a = sess.run(all_ops, feed_dict={net.y: y, net.gen_zs: gen_zs, net.gen_y: gen_labels, net.ind: ind})

        max_a = np.amax(a, axis=1)
        arg_max_a = np.argmax(a, axis=1)
        correct_indicator = np.where(np.logical_and(y == arg_max_a, max_a > conf.classified_as_thresh), 1, 0)
        incorrect_indicator= np.ones(shape=correct_indicator.shape, dtype=np.int32) - correct_indicator
        # Don't report the duds, don't expect them to be classified correctly
        ones = np.ones(shape=conf.n - num_duds * conf.num_class, dtype=np.int32)
        zeros = np.zeros(shape=num_duds * conf.num_class, dtype=np.int32)
        duds_mask = np.concatenate([ones, zeros], axis=0)
        ind_of_incorrect = np.argwhere(duds_mask * incorrect_indicator)
        incorrect_responses = a[ind_of_incorrect]
        labels_of_inc = y[ind_of_incorrect]
        pos_of_inc = z[ind_of_incorrect]
        num_correct = np.sum(correct_indicator)

        sess.close()
        #tf.reset_default_graph()

    return TrainResult(num_correct, incorrect_responses, labels_of_inc, pos_of_inc,
                       conf, z_bar, tau, class_wise_z_list, z_bar_list, tau_list)


class TrainResult:

    def __init__(self, num_correct, incorrect_responses, labels_of_inc, pos_of_inc,
                 conf, final_z_bar, final_tau, class_wise_z_list=None, z_bar_list=None, tau_list=None):
        self.num_correct = num_correct
        self.incorrect_responses = incorrect_responses
        self.labels_of_inc = labels_of_inc
        self.pos_of_inc = pos_of_inc
        self.class_wise_z_list = class_wise_z_list
        self.final_z_bar = final_z_bar
        self.final_tau = final_tau
        self.z_bar_list = z_bar_list
        self.tau_list = tau_list
        self.incorr_report_limit = conf.incorr_report_limit

    def get(self):
        return self.class_wise_z_list, self.z_bar_list, self.tau_list

    def report_incorrect(self):
        num_inc = self.incorrect_responses.shape[0]
        num_to_report = min(self.incorr_report_limit, num_inc)
        for i in xrange(num_to_report):
            print str(self.labels_of_inc[i][0]) + ': ' + str(self.incorrect_responses[i])
            print self.pos_of_inc[i]
            print ""


