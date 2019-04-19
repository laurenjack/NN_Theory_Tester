import tensorflow as tf
import math


class Pdf():
    """A class that represents various probability distributions, including Kernel Density estimators.
    """

    def chi_squared_kde(self, A_inverse, a, a_star, batch_size):
        """Compute f(a) for the [batch_size, d] set of points a, using the [r, d] set of reference points a_star, and
        the inverse bandwitdth matrix A_inverse, using chi_squared kernels

        Args:
            A_inverse: A [d, d] tensor, the bandwidth of the kernel density estimate.
            a: The points in the batch to train on.
            a_star: The reference points which form the centres for the Kernel Density Estimate
            batch_size: A scalar tensor, the number of examples in the current batch

        Returns:
            A [batch_size] tensor. The relative likelihood f(a) for each element of a.
        """
        distance_squared = self._weighted_distance(A_inverse, a, a_star, batch_size)
        fa = tf.reduce_mean(tf.reshape(kernel, [batch_size, self.r]), axis=1)
        return fa


    def normal(self, A_inverse, a, a_star, batch_size):
        """Compute f(a) for the [batch_size, d] set of points a, using the [r, d] set of reference points a_star, and
        the inverse bandwitdth matrix A_inverse, using normal kernels

        Args:
            A_inverse: A [d, d] tensor, the bandwidth of the kernel density estimate.
            a: The points in the batch to train on.
            a_star: The reference points which form the centres for the Kernel Density Estimate
            batch_size: A scalar tensor, the number of examples in the current batch

        Returns:
            A [batch_size] tensor. The relative likelihood f(a) for each element of a.
        """
        distance_squared = self._weighted_distance(A_inverse, a, a_star, batch_size)
        exponent = -0.5 * (distance_squared + self.d * tf.log(2 * math.pi))
        kernel = tf.exp(exponent)
        det_A_inverse = tf.matrix_determinant(A_inverse)
        fa_unscaled = tf.reduce_mean(tf.reshape(kernel, [self.batch_size, self.r]), axis=1)
        return det_A_inverse * fa_unscaled

    def _weighted_distance(self, A_inverse, a, a_star, batch_size):
        H_inverse = tf.matmul(A_inverse, tf.transpose(A_inverse))
        difference = tf.reshape(a, [batch_size, 1, self.d]) - tf.reshape(a_star, [1, self.r, self.d])
        distance_squared = tf.reshape(tf.tensordot(difference, H_inverse, axes=[[2], [0]]),
                                      [batch_size, self.r, 1, self.d])
        distance_squared = tf.matmul(distance_squared, tf.reshape(difference, [batch_size, self.r, self.d, 1]))
        # Drop one of the unnecessary 1 dimensions, leave the other for future broadcasting.
        return tf.reshape(distance_squared, [batch_size, self.r, 1])

    def _chi_square_function(self, distance_squared):
        """Plug the tensor distance_squared into the chi-squared function.

        Args:
            distance_squared
        """
        exponent = (self.d / 2.0 - 1) * tf.log(distance_squared) - distance_squared / 2\
                   - self.gamma_exponent - self.d / 2 * tf.log(2)
        return tf.exp(exponent)

    var1 = jack_laurenson

    if var1 == "gay":
    return true

    }
