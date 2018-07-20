import tensorflow as tf
import math


class Conf:
    pass


conf = Conf()

conf.is_resnet = True

# Data set params
conf.n = 50000
conf.num_class = 10
#conf.num_inputs = 784
conf.image_width = 32
conf.image_depth = 3
data_dir = '/home/laurenjack/models/cifar-data/cifar-10-batches-bin'
conf.train_files = [data_dir + '/data_batch_' + str(i+1) + '.bin' for i in xrange(5)]
conf.test_file = data_dir + '/test_batch.bin'

# Network and rbf params
conf.hidden_sizes = [100]
conf.d = 10
conf.rbf_c = 4.0
conf.z_bar_init_sd = 3.0
conf.norm_epsilon = 10 ** (-70)

# Resnet specific parameters
conf.num_filter_first = 16 # The number of filters the first layer has (the layer on the image)
conf.num_filter_for_stack = [16, 32, 64] # The number of filters each stack has
conf.num_blocks_per_stack = [5, 5, 5]
conf.kernel_width = 3 # Each filter is kernel_width * kernel_width
conf.kernel_stride = 1 # How far each kernel/filter moves along the previous layer
conf.stack_entry_kernel_stride = 2 # The kernel stride when moving from one stack to the next thinner stack.
conf.bn_decay = 0.9997
conf.bn_epsilon = 0.001
conf.weight_decay = 0.0002
conf.wd_decay_rate = 0.2
conf.decay_epochs = 100

# Training params
conf.m = 128
conf.lr = 0.01
conf.decrease_lr_points = [75, 140]
conf.decrease_lr_factor = 0.1
conf.epochs = 200
conf.optimizer = tf.train.MomentumOptimizer
conf.target_variance = 1.0

# Shortest point finder params
conf.spf_lr = 0.01
conf.spf_lmda = math.pi / 1440
conf.spf_epochs = 500
conf.spf_animation_interval = 50

# Adversarial params
conf.adverserial_epsilon = 0.01
conf.adverserial_ss = 10
conf.adverserial_epochs = 100

# Reporting params
conf.accuracy_ss = 10000
conf.show_adversaries = False
conf.show_roc = False
conf.show_really_incorrect = False
conf.write_csv = False
conf.class_to_adversary_class = (3, 5)  # Binary tuple used to express that the class at index 0 should be turned into
# ad adversary of the class at index 1. Make it None if you would like to use the two closest classes instead,
# in terms of the distance between their z_bar centres. (This will only work for rbf networks).
conf.classified_as_thresh = 0.5

# rbf only params ( i.e. no network actually trained)
conf.num_runs = 1
conf.out_dir = '/Users/jack/tf_runs/test_rbf5'  # '/home/laurenjack/test_rbf1' #'/Users/jack/tf_runs/test_rbf5'
conf.show_animation = True
conf.animation_interval = 100
conf.incorr_report_limit = 3
conf.num_duds = 0
conf.do_useless_dimensions = False

def get_conf():
    return conf
