import tensorflow as tf


class VariableCreator():
    """
    A small class for creating Tensorflow variables.

    The reason for this classes existence is having a convenient, non-intrusive way of mocking the creation of
    variables.
    """

    def get_variable(self, name, shape, initializer, dtype=tf.float32, trainable=True):
        """A little wrapper around tf.get_variable"""
        return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, trainable=trainable)