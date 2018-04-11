import tensorflow as tf
import numpy as np
from test_rbf import RBF
from gen_train_points import *
from animator import *

class Conf:
    pass

conf = Conf()
conf.n = 100
conf.num_class = 5
conf.d = 2
conf.rbf_c = 10.0
conf.z_bar_init_sd = 3.0
conf.z_sd = 6.0
conf.lr = 0.2
conf.show_details = False
conf.show_animation = True
conf.train_centres_taus = False
conf.optimizer = tf.train.AdamOptimizer
epochs = 500

net = RBF(conf)
all_ops = net.all_ops()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
y = gen_train_labels(conf)

class_wise_z_list = []
for k in xrange(conf.num_class):
    class_wise_z_list.append([])
z_bar_list = []
tau_list = []

for e in xrange(epochs):
    _, z, z_bar, tau, a = sess.run(all_ops, feed_dict={net.y: y})
    for k in xrange(conf.num_class):
        ind_of_class = np.argwhere(y == k)[:, 0]
        class_wise_z_list[k].append(z[ind_of_class])
    z_bar_list.append(z_bar)
    tau_list.append(tau)

animate(class_wise_z_list, z_bar_list, tau_list, conf)

