import numpy as np
import tensorflow as tf

def find_shortest_point(conf, z_bar, tau):
    """Find the shortest point between a single pair of clusters"""
    d = conf.d
    spf_lr = conf.spf_lr
    spf_epochs = conf.spf_epochs

    #Compute the parameters used in the hyperbola
    z_bar_i = z_bar[:, 0]
    z_bar_j = z_bar[:, 1]
    tau_i = tau[:, 0]
    tau_j = tau[:, 1]

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

    g_2 = tf.Graph()
    with g_2.as_default():
        spf_finder = SpfFinder(d, spf_lr, z_bar_i, tau_sq_i, k, scaling_factors, num_positive, sign, [0.5, 1.0, 1.5])

        all_ops = spf_finder.all_ops()
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        for e in xrange(spf_epochs):
            z =sess.run(all_ops)

        print report_costs(z, z_bar_i, tau_sq_i, z_bar_j, tau_sq_j)
        print "Done"

def report_costs(z, z_bar_i, tau_sq_i, z_bar_j, tau_sq_j):
    C1 = np.sum(tau_sq_i * (z - z_bar_i) ** 2.0)
    C2 = np.sum(tau_sq_j * (z - z_bar_j) ** 2.0)
    return C1, C2

class SpfFinder:

    def __init__(self, d, spf_lr, z_bar_i, tau_sq_i, k, scaling_factors, num_positive, sign, theta_start):
        # z constants
        z_bar_i = tf.constant(z_bar_i, tf.float32, name='z_bar_i')
        tau_sq_i = tf.constant(tau_sq_i, tf.float32, name='tau_sq_i')

        # hyperbola constants
        k_tf = tf.constant(k, tf.float32, name='k')
        scaling_factors_tf = tf.constant(scaling_factors, tf.float32, name='scaling_factors')

        # Create the list of d-1 theta
        thetas = [tf.get_variable('theta' + str(i), [], tf.float32,
                                  initializer=tf.constant_initializer(theta_start[i])) for i in xrange(d - 1)]
        trig_prods = []

        # Build the hyperbola
        # All positive scaling factors, we have an elipse
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

        #Re order trig products according to correct sign
        pos_ind = 0
        neg_ind = num_positive
        unscaled_z = []
        for i in xrange(d):
            if sign[i] > 0:
                unscaled_z.append(trig_prods[pos_ind])
                pos_ind+=1
            else:
                unscaled_z.append(trig_prods[neg_ind])
                neg_ind+=1


        # Construct the cartesian coordinates
        self.z_unscaled = tf.stack(unscaled_z, axis=0)
        self.z = scaling_factors_tf * self.z_unscaled + k_tf

        #Build the remainder of the graph, onwards to the cost function
        # weighted_diffs = tau_sq_i * (self.z - z_bar_i) ** 2.0
        # C = tf.reduce_mean(weighted_diffs)
        # opt = tf.train.GradientDescentOptimizer(spf_lr)
        # self.train_op = opt.minimize(C)

    def all_ops(self):
        return self.z








    # # Randomly generate a point for GD
    # z = np.random.randn(d) * z_sd
    # z_list = []
    # z_list.append(z)
    #
    # for i in xrange(spf_epochs):
    #     dC_dz, dReg_dz = _compute_gradient(z, z_bar_i, z_bar_j, tau_i, tau_j, d)
    #     z_new = z + dC_dz * spf_lr + dReg_dz * spf_lmda
    #     z = z_new
    #     z_list.append(z)
    #
    # return z_list



def _secant(theta):
    """Prodcue the ith secant"""
    cos = tf.cos(theta)
    sec = 1.0 / cos
    return sec

def _apply_circle_prod(trig_prods, thetas, start, end, theta_off_set=0):
    for i in xrange(start, end-1):
        theta = thetas[i+theta_off_set]
        trig_prods[i] *= tf.cos(theta)
        sin_theta = tf.sin(theta)
        for j in xrange(i+1, end):
            trig_prods[j] *= sin_theta
    return trig_prods

# def _compute_gradient(theta, z_bar_i, z_bar_j, tau_i, tau_j, d):
#     #Compute distances
#     z_diff_i = z - z_bar_i
#     z_diff_j = z - z_bar_j
#     dist_i = np.sum(z_diff_i ** 2.0)
#     dist_j = np.sum(z_diff_j ** 2.0)
#
#     dC_dz = 2.0 * z_diff_i
#
#     tau_sq_i = tau_i ** 2.0
#     tau_sq_j = tau_j ** 2.0
#     tau_sq_diff = tau_sq_i - tau_sq_j
#     weighted_centre_diff = tau_sq_i * z_bar_i - tau_sq_j * z_bar_j
#     k = weighted_centre_diff / tau_sq_diff
#     complete_sq_constants = weighted_centre_diff ** 2.0 / tau_sq_diff
#     quad_constants = tau_j * tau_sq_j ** 2.0 - tau_sq_i * z_bar_i ** 2.0
#     total_constant = np.sum(complete_sq_constants + quad_constants)
#     scaling_factors = (total_constant / tau_sq_diff) ** 0.5
#
#     # switch the sides if the total constant K is negative, eg:
#     # c1 * x^2 - c2 * y^2 - c^3 * z^2 = - K
#     # becomes
#     # - c1 * x^2 + c2 * y^2 + c^3 * z^2 = K
#     # Where c1, c2, c3 and K are positive constants
#
#
#     #Compute z and theta's derivatives
#     sin = np.sin(theta)
#     cos = np.cos(theta)
#
#
#
#     first  = cos[:1]
#     all_but_last = sin[:-1]
#     all_but_first = cos[1:]
#     last = sin[-1:]
#
#     unscaled_z = np.concatenate([first, all_but_last - all_but_first, last])
#     z = scaling_factors * (unscaled_z + k)
#
#     d_sin_all = cos
#     d_neg_cos_all = sin
#     #Switch sign of the first element as its positive cosine
#     d_neg_cos_all[0] = -d_neg_cos_all[0]
#
#     dC_d_sin =  dC_dz[:-1] * d_sin_all
#     dC_d_neg_cos = dC_dz[1:] * d_neg_cos_all
#     dC_d_theta = dC_d_sin + dC_d_neg_cos
#     return dC_d_theta





