import numpy as np
import configuration
conf = configuration.get_configuration()

"""Responsible for adversarial operations"""

# TODO(Jack) include in docs properly
""" Given an image and a deliberately faulty label, return the gradient
        with respect to that image which minimises the loss function, using that
        faulty label.

        As the z grads ignore off target points, simply trying to
        maximise the loss function will just run the example down into a low
        rbf region. Therefore, rather than maximising the loss, we minimise
        the loss with respect to a different target. Ideally the nearest z_bar"""

def adversarial_gd(network_runner, correct, closest_classes):
    """Use gradient descent to generate adversarial examples"""
    x, adversarial_y, actual_y = _select_sample(correct, closest_classes)
    adverse_op = network_runner.network.gradient_wrt_inputs()
    x_orig = x

    for e in xrange(conf.adversarial_epochs):
        dx = network_runner.feed_and_run(x, adversarial_y, adverse_op)
        # normed =  dx / np.sum(dx ** 2.0) ** 0.5
        normed = np.sign(dx)
        x = x - conf.adversarial_epsilon * normed

    return x, actual_y, x_orig


def _select_sample(correct, closest_classes):
    k1, k2 = closest_classes
    corr_k1 = correct.get_sample_of_class(k1, conf.adversarial_ss)
    x = corr_k1.x
    actual_y = corr_k1.y
    num_adverse = x.shape[0]
    adversarial_y = k2 * np.ones(num_adverse)
    return x, adversarial_y, actual_y