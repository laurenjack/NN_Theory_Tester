from shortest_point_finder import *
import numpy as np


class Conf:
    pass
conf = Conf()

conf.d = 4
conf.spf_lr = 0.1
conf.spf_epochs = 1

#Test z bars
z_bar = np.array([[-1.0, -1.5, 1.5, 1.0], [2.0, 1.0, 3.0, -1.5]]).transpose()
tau = np.array([[1.0, 0.5, 0.25, 2.0], [0.5, 1.0, 1.5, 3.0]]).transpose()

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

#if total_constant < 0:
#    scaling_factors_pre_root = (-scaling_factors_pre_root)

sign = np.sign(scaling_factors_pre_root)
scaling_factors = sign * abs(scaling_factors_pre_root) ** 0.5


#Test the mathematics
z = np.array([0.0, 0.16666667, 2.785714286, -0.5])
print report_costs(z, z_bar_i, tau_sq_i, z_bar_j, tau_sq_j)

should_be_1 = np.sum(1.0 / scaling_factors_pre_root * (z - k) ** 2.0)
print should_be_1

should_be_0 = np.sum(tau_sq_diff * (z - weighted_centre_diff/tau_sq_diff) ** 2.0 -complete_sq_constants + quad_constants)
print should_be_0

print total_constant
print np.sum(tau_sq_diff * (z - weighted_centre_diff/tau_sq_diff) ** 2.0)


find_shortest_point(conf, z_bar, tau)
