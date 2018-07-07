import numpy as np
import tensorflow as tf
import math

def find_shortest_point(conf, z_bar, tau):
    """Find the shortest point between a single pair of clusters"""
    d = conf.d
    spf_lr = conf.spf_lr
    spf_epochs = conf.spf_epochs

    #Compute the parameters used in the hyperbola
    i, j = _quick_shortest(z_bar)
    z_bar_i = z_bar[:, i]
    z_bar_j = z_bar[:, j]
    tau_i = tau[:, i]
    tau_j = tau[:, j]

    tau_sq_i = tau_i ** 2.0
    tau_sq_j = tau_j ** 2.0
    tau_sq_diff = tau_sq_i - tau_sq_j
    weighted_centre_diff = tau_sq_i * z_bar_i - tau_sq_j * z_bar_j
    k = weighted_centre_diff / tau_sq_diff
    complete_sq_constants = weighted_centre_diff ** 2.0 / tau_sq_diff
    quad_constants = tau_sq_i * z_bar_i ** 2.0 - tau_sq_j * z_bar_j ** 2.0
    total_constant = np.sum(complete_sq_constants - quad_constants)
    scaling_factors_pre_root = total_constant / tau_sq_diff

    # if total_constant < 0:
    #     scaling_factors_pre_root = (-scaling_factors_pre_root)

    sign = np.sign(scaling_factors_pre_root)
    scaling_factors = abs(scaling_factors_pre_root) ** 0.5

    # Count the number of positive scaling factors
    num_positive = np.sum(np.greater(sign, 0).astype(np.int32))
    # Should not be negative, would be be absurd, must be a bug
    if num_positive == 0:
        raise RuntimeError("Mathematically impossible, must be a bug")

    g_spf = tf.Graph()
    with g_spf.as_default():
        init_theta_a = np.random.uniform(0.0, math.pi / 2.0 - 0.1, d)
        init_theta_b = np.random.uniform(math.pi / 2.0 + 0.1, 3.0 * math.pi / 2.0 - 0.1, d)
        spf_finder = SpfFinder(d, spf_lr, z_bar_i, tau_sq_i, k, scaling_factors, num_positive,
                               sign, init_theta_a, init_theta_b, conf.rbf_c)

        ops_a, ops_b = spf_finder.all_ops()
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        z_list = []
        for e in xrange(spf_epochs):
            _, theta_a, z_a, C_a, rbf_a =sess.run(ops_a)
            _, theta_b, z_b, C_b, rbf_b = sess.run(ops_b)
            z_list.append(np.array([z_a, z_b]))


    return z_list, [C_a, C_b], [rbf_a, rbf_b], (z_bar_i, z_bar_j), (tau_i, tau_j), (i, j)

def report_costs(z, z_bar_i, tau_sq_i, z_bar_j, tau_sq_j):
    C1 = np.sum(tau_sq_i * (z - z_bar_i) ** 2.0)
    C2 = np.sum(tau_sq_j * (z - z_bar_j) ** 2.0)
    return C1, C2

class SpfFinder:

    def __init__(self, d, spf_lr, z_bar_i, tau_sq_i, k, scaling_factors, num_positive,
                 sign, init_theta_a, init_theta_b, rbf_c):
        self.ops_a = self._create_graph('theta_a', init_theta_a, d, num_positive, scaling_factors, k, sign,
                                   z_bar_i, tau_sq_i, spf_lr, rbf_c)
        self.ops_b = self._create_graph('theta_b', init_theta_b, d, num_positive, scaling_factors, k, sign,
                                   z_bar_i, tau_sq_i, spf_lr, rbf_c)


    def all_ops(self):
        return self.ops_a, self.ops_b

    def _create_thethas(self, prefix, init_theta, d):
        [tf.get_variable(prefix + str(i), [], tf.float32,
                         initializer=tf.constant_initializer(init_theta[i])) for i in xrange(d - 1)]

    def _create_graph(self, theta_prefix, theta_init, d, num_positive, scaling_factors, k, sign,
                      z_bar_i, tau_sq_i, spf_lr, rbf_c):

        thetas = [tf.get_variable(theta_prefix + str(i), [], tf.float32,
                         initializer=tf.constant_initializer(theta_init[i])) for i in xrange(d - 1)]

        # Build the hyperbola
        # All positive scaling factors, we have an elipse
        trig_prods = []
        if num_positive == d:
            trig_prods = [1.0 for i in xrange(d)]
            _apply_circle_prod(trig_prods, thetas, 0, d)
        else:
            for i in xrange(num_positive):
                trig_prods.append(_secant(thetas[0]))
            _apply_circle_prod(trig_prods, thetas, 0, num_positive, theta_off_set=1)
            for i in xrange(num_positive, d):
                trig_prods.append(tf.tan(thetas[0]))
            _apply_circle_prod(trig_prods, thetas, num_positive, d)

        # Re order trig products according to correct sign
        pos_ind = 0
        neg_ind = num_positive
        unscaled_z = []
        for i in xrange(d):
            if sign[i] > 0:
                unscaled_z.append(trig_prods[pos_ind])
                pos_ind += 1
            else:
                unscaled_z.append(trig_prods[neg_ind])
                neg_ind += 1

        # Construct the cartesian coordinates
        z_unscaled = tf.stack(unscaled_z, axis=0)
        z = scaling_factors * z_unscaled + k

        # Build the remainder of the graph, onwards to the cost function
        weighted_diffs = tau_sq_i * (z - z_bar_i) ** 2.0
        C = tf.reduce_mean(weighted_diffs)
        # opt = tf.train.AdamOptimizer(spf_lr)
        opt = tf.train.GradientDescentOptimizer(spf_lr)
        grads = opt.compute_gradients(C)
        new_grads = []
        for grad, var in grads:
            if grad is not None:
                grad = tf.sign(grad)
            new_grads.append((grad, var))
        train_op = opt.apply_gradients(new_grads)
        #train_op = opt.minimize(C)
        #Report rbf
        exp = tf.exp(-C)
        #rbf = rbf_c * exp
        return train_op, thetas, z, C, exp


def _secant(theta):
    """Prodcue the ith secant"""
    cos = tf.cos(theta)
    sec = 1.0 / cos
    return sec

def _apply_circle_prod(trig_prods, thetas, start, end, theta_off_set=0):
    running_chain = 1.0
    for i in xrange(start, end-1):
        theta = thetas[i + theta_off_set]
        trig_prods[i] *= tf.cos(theta) * running_chain
        sin_theta = tf.sin(theta)
        running_chain *= sin_theta
    #Last dimension, all sines
    trig_prods[-1] *= running_chain

    return trig_prods

def _quick_shortest(z_bar):
    dists = []
    inds = []
    K = z_bar.shape[1]
    for i in xrange(K):
        zb1 = z_bar[:, i]
        for j in xrange(i+1, K, 1):
            zb2 = z_bar[:, j]
            if i != j:
                dists.append(np.sum((zb1 - zb2) ** 2.0))
                inds.append((i, j))
    dists = np.array(dists)
    ind = np.argmin(dists)
    return inds[ind]








