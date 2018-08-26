import numpy as np
from configuration import conf

"""Responsible for adverserial operations"""

def adverserial_gd(network_runner, correct, closest_classes):
    """Use gradient descent to generate adverserial examples"""
    x, adverserial_y, actual_y = _select_sample(correct, closest_classes)
    adverse_op = network_runner.network.adverserial_gradient()
    x_orig = x

    for e in xrange(conf.adverserial_epochs):
        dx = network_runner.feed_and_run(x, adverserial_y, adverse_op)
        # normed =  dx / np.sum(dx ** 2.0) ** 0.5
        normed = np.sign(dx)
        x = x - conf.adverserial_epsilon * normed

    return x, actual_y, x_orig


def adverserial_fgsm(network_runner, correct, closest_classes):
    x, adverserial_y, actual_y = _select_sample(correct, closest_classes)
    adverse_op = network_runner.network.fgsm_adverserial_with_target()
    x = network_runner.feed_and_run(x, adverserial_y, adverse_op)
    return network_runner, x, actual_y

def _select_sample(correct, closest_classes):
    k1, k2 = closest_classes
    corr_k1 = correct.get_sample_of_class(k1, conf.adverserial_ss)
    x = corr_k1.x
    actual_y = corr_k1.y
    num_adverse = x.shape[0]
    adverserial_y = k2 * np.ones(num_adverse)
    return x, adverserial_y, actual_y