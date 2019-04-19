import tensorflow as tf


class KernelFunctions():
    """A class that represents that produces Kernel Density estimators based on various Kernel functions.
    """

    def pdf(self, A_inverse, a, a_star, batch_size):
        """Compute f(a) for the [batch_size, d] set of points a, using the [r, d] set of reference points and the
        inverse bandwitdth matrix A_inverse.

        Args:
            A_inverse: A [d, d] tensor, the bandwidth of the kernel density estimate.

        Returns:
            A [batch_size] tensor. The relative likelihood f(a) for each element of a.
        """
        H_inverse = tf.matmul(A_inverse, tf.transpose(A_inverse))
        difference = tf.reshape(self.a, [self.batch_size, 1, self.d]) - tf.reshape(a_star, [1, self.r, self.d])
        distance_squared = tf.reshape(tf.tensordot(difference, H_inverse, axes=[[2], [0]]),
                                      [self.batch_size, self.r, 1, self.d])
        distance_squared = tf.matmul(distance_squared, tf.reshape(difference, [self.batch_size, self.r, self.d, 1]))
        # Drop one of the unnecessary 1 dimensions, leave the other for future broadcasting.
        distance_squared = tf.reshape(distance_squared, [self.batch_size, self.r, 1])
        exponent = 0.5 * (-distance_squared + self.d)
        kernel = tf.exp(exponent)
        det_A_inverse = tf.matrix_determinant(A_inverse)
        fa_unscaled = tf.reduce_mean(tf.reshape(kernel, [self.batch_size, self.r]), axis=1)
        return det_A_inverse * fa_unscaled  # / (2.0 * math.pi) ** (self.d * 0.5)