import tensorflow as tf
import numpy as np

from typing import List


def max_detection(soft_mask, window_size: List, threshold: float):

    sm_shape = soft_mask.get_shape().as_list()
    n_dim = len(sm_shape) - 2
    sm_dims = [sm_shape[i + 1] for i in range(n_dim)]
    w_dims = [window_size[i + 1] for i in range(n_dim)]
    w_dims_padded = [1, *w_dims, 1]

    if n_dim == 2:
        data_format = "NHWC"
        pool_func = tf.nn.max_pool2d
        conv_transpose = tf.nn.conv2d_transpose
    elif n_dim == 3:
        data_format = "NDHWC"
        pool_func = tf.nn.max_pool3d
        conv_transpose = tf.nn.conv3d_transpose

    max_pool = pool_func(
        soft_mask, w_dims_padded, w_dims_padded, padding="SAME", data_format=data_format
    )

    conv_filter = np.ones([*w_dims, 1, 1])

    upsampled = conv_transpose(
        max_pool,
        conv_filter.astype(np.float32),
        [1, *sm_dims, 1],
        w_dims_padded,
        padding="SAME",
        data_format=data_format,
        name="nms_conv_0",
    )

    maxima = tf.equal(upsampled, soft_mask)
    threshold_maxima = tf.logical_and(maxima, soft_mask >= threshold)
    maxima = threshold_maxima

    # Fix doubles
    # Check the necessary window size and adapt for isotropic vs unisotropic nms:
    double_suppresion_window = [1, *[1 if dim == 1 else 3 for dim in w_dims], 1]
    # add 1 to all local maxima
    sm_maxima = tf.add(tf.cast(maxima, tf.float32), soft_mask)

    # sm_maxima smoothed over large window
    max_pool = pool_func(
        sm_maxima,
        double_suppresion_window,
        [1 for _ in range(n_dim + 2)],
        padding="SAME",
        data_format=data_format,
    )

    # not sure if this does anything
    conv_filter = np.ones([1 for _ in range(n_dim + 2)])
    upsampled = conv_transpose(
        max_pool,
        conv_filter.astype(np.float32),
        [1, *sm_dims, 1],
        [1 for _ in range(n_dim + 2)],
        padding="SAME",
        data_format=data_format,
        name="nms_conv_1",
    )

    reduced_maxima = tf.equal(upsampled, sm_maxima)
    reduced_maxima = tf.logical_and(reduced_maxima, sm_maxima > 1)
    reduced_maxima = tf.reshape(reduced_maxima, sm_shape)

    if n_dim == 2:
        return (
            tf.reshape(maxima, sm_dims, name="maxima"),
            tf.reshape(reduced_maxima, sm_dims, name="reduced_maxima"),
        )
    elif n_dim == 3:
        return (
            tf.reshape(maxima, sm_dims, name="maxima"),
            tf.reshape(reduced_maxima, sm_dims, name="reduced_maxima"),
        )
