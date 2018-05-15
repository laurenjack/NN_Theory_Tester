import tensorflow as tf
from animator import *
from train_rbf import train
from shortest_point_finder import find_shortest_point

class Conf:
    pass
conf = Conf()

#meta_params
conf.num_runs = 1
conf.out_dir = '/home/laurenjack/test_rbf1' #'/Users/jack/tf_runs/test_rbf5'
conf.show_animation = True
conf.animation_interval = 200
conf.incorr_report_limit = 3

conf.n = 100
conf.num_class = 5
conf.d = 2
conf.rbf_c = 1.0
conf.z_bar_init_sd = 3.0
conf.z_sd = 6.0
conf.lr = 0.1
conf.show_details = False
conf.train_centres_taus = True
conf.epochs = 300
conf.classified_as_thresh = 0.2
conf.optimizer = tf.train.GradientDescentOptimizer
conf.start_reg = 150

conf.spf_lr = 0.01
conf.spf_lmda = 0.1
conf.spf_epochs = 500
conf.spf_animation_interval = 200

total_correct = 0
for i in xrange(conf.num_runs):
    train_result = train(conf)
    train_result.report_incorrect()
    num_correct = train_result.num_correct
    print float(num_correct) / float(conf.n) * 100
    total_correct += num_correct
    z_bar = train_result.final_z_bar
    tau = train_result.final_tau
    sp_z_list, Cs, rbfs = find_shortest_point(conf, z_bar, tau)
    print Cs
    print rbfs
print ""
tpc = float(total_correct) / float(conf.n * conf.num_runs) * 100
print "Total Percentage Correct: "+str(tpc)


#Take the last a and evaluate the percentage of correctly classified points

if conf.num_runs == 1 and conf.d == 2:
    animate(train_result, conf)
    animate_spf(z_bar, tau, sp_z_list, conf)

