
# coding: utf-8

import numpy as np
import tensorflow as tf


def xavier_weight_init(shape):
    """Returns function that creates random tensor.

    The specified function will take in a shape (tuple or 1-d array) and
    returns a random tensor of the specified shape drawn from the
    Xavier initialization distribution.

    Hint: You might find tf.random_uniform useful.
    """
   
    sum_shape = sum(list(shape))
    epsilon = tf.sqrt(6.0/sum_shape)
    return tf.random_uniform(shape=shape, minval=-epsilon, maxval=epsilon)


if __name__ == "__main__":
    test_initialization_basic()

