import tensorflow as tf

import src.rbf_softmax.rbf as rb
from src.rbf_softmax.gen_train_points import *

conf = configuration.get_configuration()

def train():
    g_1 = tf.Graph()
    with g_1.as_default():
        with tf.device('/cpu:0'):
            n = conf.n
            m = conf.m
            z_init = tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd)
            batch_inds_ph = tf.placeholder(tf.int32, shape=[None], name="batch_inds")
            z_var = tf.get_variable("z", shape=[conf.n, conf.d], initializer=z_init)
            z_bar_init = tf.truncated_normal_initializer(stddev=conf.z_bar_init_sd)
            tau_init = tf.constant_initializer(0.5 / 1.0 ** 0.5 * np.ones(shape=[conf.d, conf.num_class])) # float(conf.d)
            net = rb.Rbf(z_bar_init, tau_init)
            rbf_ops = net.create_all_ops(z_var, batch_inds_ph)
            core_ops = rbf_ops.core_ops()

            #Add training op
            core_ops[0] = tf.train.MomentumOptimizer(learning_rate=conf.lr, momentum=0.9).minimize(core_ops[0])

            # Summaries for variables
            if conf.num_runs == 1:
                for var in tf.trainable_variables():
                    zeros = tf.zeros(shape=var.shape)
                    tf.summary.histogram(var.op.name, tf.where(tf.is_nan(var), zeros, var))
                summary_op = tf.summary.merge_all()
                core_ops.append(summary_op)

            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            y = gen_train_labels()

            if conf.num_runs == 1:
                summary_writer = tf.summary.FileWriter(conf.rbf_only_out_dir , sess.graph)

            class_wise_z_list = None
            z_bar_list = None
            tau_list = None
            if conf.num_runs == 1:
                class_wise_z_list = []
                for k in xrange(conf.num_class):
                    class_wise_z_list.append([])
                z_bar_list = []
                tau_list = []

            batch_indicies = np.arange(conf.n)
            all_n = np.arange(conf.n)
            first_d = np.zeros(conf.n, dtype=np.int32)
            for e in xrange(conf.epochs):
                np.random.shuffle(batch_indicies)
                if conf.num_runs == 1:
                    for k in xrange(0, n, m):
                        batch = batch_indicies[k:k + m]
                        batch_size = batch.shape[0]
                        feed_dict = {net.y: y[batch], rb.batch_size: batch_size, batch_inds_ph: batch}
                        _, z, z_bar, tau, a, rbf_prob, weighted_variance, z_diff, neg_dist, \
                        weighted_z_diff_sq, summ_str = sess.run(core_ops, feed_dict=feed_dict)

                    feed_dict = {net.y: y, batch_inds_ph: all_n, rb.batch_size: conf.n}
                    z, tau, rbf_prob, weighted_variance, z_diff, neg_dist, weighted_z_diff_sq = sess.run((core_ops[1], core_ops[3], core_ops[5], core_ops[6],
                                                                  core_ops[7], core_ops[8], core_ops[9]), feed_dict=feed_dict)
                    z_diff_just_class = abs(z_diff[all_n, first_d, y])
                    #weighted_z_diff_sq_just_class = abs(weighted_z_diff_sq[all_n, first_d, y])
                    for k in xrange(conf.num_class):
                        just_k = z_diff_just_class[y == k]
                        weighted_z_diff_sq_k = np.mean(weighted_z_diff_sq[y == k][:, :, k],axis=1)
                        z_diff_just_class_avg = np.mean(just_k)
                        tau_0 = tau[0, k]
                        rbf_prob_k = rbf_prob[y == k][:, k]
                        neg_dist_k = neg_dist[y == k][:, k]
                        weighted_var_exp_0 = np.sum(tau_0 ** 2.0 * z_diff_just_class_avg ** 2.0) / 10.0
                        weighted_variance_0 = weighted_variance[0, k]
                        ind_of_class = np.argwhere(y == k)[:, 0]
                        class_wise_z_list[k].append(z[ind_of_class])
                    z_bar_list.append(z_bar)
                    tau_list.append(tau)
                    summary_writer.add_summary(summ_str, e)
                else:
                    for k in xrange(0, n, m):
                        batch = batch_indicies[k:k + m]
                        batch_size = batch.shape[0]
                        feed_dict = {net.y: y[batch], rb.batch_size: batch_size, batch_inds_ph: batch}
                    _, z, z_bar, tau, a = sess.run(core_ops, feed_dict=feed_dict)


            feed_dict = {net.y: y, rb.batch_size: conf.n, batch_inds_ph: np.arange(n)}
            a = sess.run(core_ops[4], feed_dict=feed_dict)
            max_a = np.amax(a, axis=1)
            arg_max_a = np.argmax(a, axis=1)
            correct_indicator = np.where(np.logical_and(y == arg_max_a, max_a > 0.5), 1, 0)
            incorrect_indicator= np.ones(shape=correct_indicator.shape, dtype=np.int32) - correct_indicator
            ind_of_incorrect = np.argwhere(incorrect_indicator)
            incorrect_responses = a[ind_of_incorrect]
            labels_of_inc = y[ind_of_incorrect]
            pos_of_inc = z[ind_of_incorrect]
            num_correct = np.sum(correct_indicator)

            sess.close()
        #tf.reset_default_graph()

    return TrainResult(num_correct, incorrect_responses, labels_of_inc, pos_of_inc,
                       z_bar, tau, class_wise_z_list, z_bar_list, tau_list)


class TrainResult:

    def __init__(self, num_correct, incorrect_responses, labels_of_inc, pos_of_inc,
                 final_z_bar, final_tau, class_wise_z_list=None, z_bar_list=None, tau_list=None):
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


