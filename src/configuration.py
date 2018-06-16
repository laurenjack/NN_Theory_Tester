import tensorflow as tf
import math


class Conf:
    pass


conf = Conf()

# meta_params
conf.num_runs = 1
conf.out_dir = '/Users/jack/tf_runs/test_rbf5'  # '/home/laurenjack/test_rbf1' #'/Users/jack/tf_runs/test_rbf5'
conf.show_animation = True
conf.animation_interval = 100
conf.incorr_report_limit = 3
conf.accuracy_ss = 1000

conf.n = 55000
conf.m = 64
conf.num_class = 10
conf.d = 2
conf.rbf_c = 4.0
conf.z_bar_init_sd = 3.0
conf.z_sd = 6.0
conf.lr = 0.002
conf.show_details = False
conf.train_centres_taus = True
conf.epochs = 30
conf.classified_as_thresh = 0.5
conf.optimizer = tf.train.GradientDescentOptimizer
conf.target_variance = 0.5

#Network params
conf.num_inputs = 784
conf.hidden_sizes = [100, 100]

# BackProp Params
conf.norm_epsilon = 10 ** (-70)
conf.num_duds = 0
conf.do_useless_dimensions = False

conf.spf_lr = 0.01
conf.spf_lmda = math.pi / 1440
conf.spf_epochs = 500
conf.spf_animation_interval = 50



def get_conf():
    return conf
