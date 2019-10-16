import numpy as np
import tensorflow as tf

import data_generator as dg
import src.random_behavior as rb
import pdf_functions as pf
import distribution_configuration

def run():
    # Config
    conf = _conf()

    # Dependency Injection
    random = rb.Random()
    data_generator = dg.GaussianMixture(conf, random)

    m = conf.m
    d = conf.d
    n = conf.n
    lr = conf.lr
    epochs = conf.epochs
    a = tf.placeholder(dtype=tf.float32, shape=[None, d])
    single_mean = conf.means
    # Actual paramters to fit to
    Q_act = data_generator.Q
    lam_inv_act = data_generator.lam_inv
    px_act = pf.normal_seperate(a, single_mean, lam_inv_act, Q_act)

    # Create Model px that we must train
    Q_val = dg._random_orthogonal_matrix(d) + 0.001
    lam_inv_val = np.ones(d, dtype=np.float32)
    Q = tf.placeholder(shape=[d, d], dtype=tf.float32)
    lam_inv = tf.placeholder(shape=[d], dtype=tf.float32)
    px = pf.normal_seperate(a, single_mean, lam_inv, Q)

    # Specify complete loss with Orthogonal regularisation
    should_be_I = tf.matmul(tf.transpose(Q), Q)
    loss = tf.reduce_mean((px - px_act) ** 2)
    reg = tf.reduce_mean((should_be_I - np.eye(d)) ** 2)
    dl_dQ = tf.gradients(loss, Q)[0]
    dl_dlam_inv = tf.gradients(loss, lam_inv)[0]
    dr_dQ = tf.gradients(reg, Q)[0]
    # reg = tf.train.GradientDescentOptimizer(lr).minimize(loss)


    # Generate some Data
    X = data_generator.sample(n)

    # Training Setup
    session = tf.Session()
    tf.global_variables_initializer().run(session=session)

    #Intial Report
    feed_dict = {a:X[0:m], Q: Q_val, lam_inv: lam_inv_val}
    _report(session.run([px_act, px, should_be_I], feed_dict=feed_dict), Q_val, lam_inv_val)

    #Train
    indices = np.arange(n)
    for e in xrange(epochs):
        random.shuffle(indices)
        if e in conf.reduce_lr_epochs:
            lr *= 0.1
        for k in xrange(0, n, m):
            x = X[indices[k:k+m]]
            feed_dict = {a: x, Q: Q_val, lam_inv: lam_inv_val}
            current_values = session.run([dl_dQ, dr_dQ, dl_dlam_inv, px_act, px, should_be_I], feed_dict=feed_dict)
            dl_dQ_val = current_values[0]
            dr_dQ_val = current_values[1]
            dl_dlam_inv_val = current_values[2]
            dl_dQ_norm = dl_dQ_val / np.linalg.norm(dl_dQ_val)
            dr_dQ_norm = dr_dQ_val / np.linalg.norm(dr_dQ_val)
            dk_dlam_inv_norm = dl_dlam_inv_val / np.linalg.norm(dl_dlam_inv_val)
            grad = 0.49 * dl_dQ_norm + 0.51 * dr_dQ_norm
            grad /= np.linalg.norm(grad)
            Q_val -= lr*grad
            lam_inv_val -= lr*dk_dlam_inv_norm
            _report(current_values[3:], Q_val, lam_inv_val)

    print 'Actual EigenValues: {}'.format(1.0 / lam_inv_act)
    print 'Actual Q: {}'.format(Q_act)



def _report(to_report, Q_val, lam_inv_val):
    px_act, px, should_be_I = to_report
    print 'p(x) actual: {}   p(x): {}\n'.format(px_act[0], px[0])
    print 'Eigenvalues: {}'.format(1.0 / lam_inv_val)
    print 'Q: {}\n'.format(Q_val)
    print 'QtQ: {}\n\n'.format(should_be_I)


def _conf():
    class Configuration:

        def __init__(self):
            self.n = 300
            self.m = 30
            self.d = 2
            self.epochs = 100
            self.lr = 0.1
            self.reduce_lr_epochs = [40, 70]
            self.means = np.zeros((1, self.d), dtype=np.float32)
            self.fixed_A = None #np.array([[1.5, 0, 0], [0, 3.0, 0], [0, 0, 0.2]])
            self.min_eigenvalue = 0.5
            self.max_eigenvalue = 2.0

    conf = Configuration()
    return conf

if __name__ == '__main__':
    run()