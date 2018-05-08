import tensorflow as tf
from animator import *
from train_rbf import train
from shortest_point_finder import find_shortest_point

class Conf:
    pass
conf = Conf()

#meta_params
conf.num_runs = 1
conf.out_dir = '/Users/jack/tf_runs/test_rbf5' #'/home/laurenjack/test_rbf1'
conf.show_animation = True
conf.animation_interval=100
conf.incorr_report_limit = 3

conf.n = 100
conf.num_class = 5
conf.d = 2
conf.rbf_c = 3.0
conf.z_bar_init_sd = 3.0
conf.z_sd = 6.0
conf.lr = 0.1
conf.show_details = False
conf.train_centres_taus = True
conf.epochs = 300
conf.classified_as_thresh = 0.2
conf.optimizer = tf.train.GradientDescentOptimizer

conf.spf_lr = 0.1
conf.spf_lmda = 0.1
conf.spf_epochs = 300
conf.spf_animation_interval = 200

total_correct = 0
for i in xrange(conf.num_runs):
    train_result = train(conf)
    train_result.report_incorrect()
    num_correct = train_result.num_correct
    print float(num_correct) / float(conf.n) * 100
    total_correct += num_correct
print ""
tpc = float(total_correct) / float(conf.n * conf.num_runs) * 100
print "Total Percentage Correct: "+str(tpc)

#Take the last a and evaluate the percentage of correctly classified points

if conf.num_runs == 1 and conf.d == 2:
    z_bar = train_result.z_bar_list[-1]
    tau = train_result.tau_list[-1]
    animate(train_result, conf)
    sp_z_list, Cs, rbfs = find_shortest_point(conf, z_bar, tau)
    print Cs
    print rbfs
    animate_spf(z_bar, tau, sp_z_list, conf)

