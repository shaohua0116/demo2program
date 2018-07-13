import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
from util import log


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def bn_act(input, is_train, batch_norm=True, activation_fn=None, name="bn_act"):
    with tf.variable_scope(name):
        _ = input
        if activation_fn is not None:
            _ = activation_fn(_)
        if batch_norm is True:
            _ = tf.contrib.layers.batch_norm(
                _, center=True, scale=True, decay=0.9,
                is_training=is_train, updates_collections=None
            )
    return _


def conv2d(input, output_shape, is_train, info=False, k_h=4, k_w=4, s=2,
           stddev=0.01, name="conv2d", activation_fn=lrelu, batch_norm=True):
    with tf.variable_scope(name):
        _ = slim.conv2d(input, output_shape, [k_h, k_w], stride=s, activation_fn=None)
        _ = bn_act(_, is_train, batch_norm=batch_norm, activation_fn=activation_fn)
        if info: log.info('{} {}'.format(name, _))
    return _


def residual_block(input, output_shape, is_train, info=False, k=3, s=1,
                   name="residual", activation_fn=lrelu, batch_norm=True):
    with tf.variable_scope(name):
        with tf.variable_scope('res1'):
            _ = conv2d(input, output_shape, is_train, k_h=k, k_w=k, s=s,
                       activation_fn=activation_fn, batch_norm=batch_norm)
        with tf.variable_scope('res2'):
            _ = conv2d(input, output_shape, is_train, k_h=k, k_w=k, s=s,
                       activation_fn=None, batch_norm=batch_norm)
        _ = activation_fn(_ + input)
    if info: log.info('{} {}'.format(name, _))
    return _


def deconv2d(input, deconv_info, is_train, name="deconv2d", info=False,
             stddev=0.01, activation_fn=tf.nn.relu, batch_norm=True):
    with tf.variable_scope(name):
        output_shape = deconv_info[0]
        k = deconv_info[1]
        s = deconv_info[2]
        _ = layers.conv2d_transpose(
            input,
            num_outputs=output_shape,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.zeros_initializer(),
            kernel_size=[k, k], stride=[s, s], padding='SAME'
        )
        _ = bn_act(_, is_train, batch_norm=batch_norm, activation_fn=activation_fn)
        if info: log.info('{} {}'.format(name, _))
    return _


def bilinear_deconv2d(input, deconv_info, is_train, name="bilinear_deconv2d",
                      info=False, activation_fn=tf.nn.relu, batch_norm=True):
    with tf.variable_scope(name):
        output_shape = deconv_info[0]
        k = deconv_info[1]
        s = deconv_info[2]
        h = int(input.get_shape()[1]) * s
        w = int(input.get_shape()[2]) * s
        _ = tf.image.resize_bilinear(input, [h, w])
        _ = conv2d(_, output_shape, is_train, k_h=k, k_w=k, s=1,
                   batch_norm=False, activation_fn=None)
        _ = bn_act(_, is_train, batch_norm=batch_norm, activation_fn=activation_fn)
        if info: log.info('{} {}'.format(name, _))
    return _


def nn_deconv2d(input, deconv_info, is_train, name="nn_deconv2d",
                info=False, activation_fn=tf.nn.relu, batch_norm=True):
    with tf.variable_scope(name):
        output_shape = deconv_info[0]
        k = deconv_info[1]
        s = deconv_info[2]
        h = int(input.get_shape()[1]) * s
        w = int(input.get_shape()[2]) * s
        _ = tf.image.resize_nearest_neighbor(input, [h, w])
        _ = conv2d(_, output_shape, is_train, k_h=k, k_w=k, s=1,
                   batch_norm=False, activation_fn=None)
        _ = bn_act(_, is_train, batch_norm=batch_norm, activation_fn=activation_fn)
        if info: log.info('{} {}'.format(name, _))
    return _


def transpose_deconv3d(input, deconv_info, is_train=True, name="deconv3d",
                       stddev=0.01, activation_fn=tf.nn.relu, batch_norm=True):
    with tf.variable_scope(name):
        output_shape = deconv_info[0]
        k = deconv_info[1]
        s = deconv_info[2]
        _ = tf.layers.conv3d_transpose(
            input,
            filters=output_shape,
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            bias_initializer=tf.zeros_initializer(),
            kernel_size=[k, k, k], strides=[s, s, s], padding='SAME'
        )
        _ = bn_act(_, is_train, batch_norm=batch_norm, activation_fn=activation_fn)
    return _


def residual_conv(input, num_filters, filter_size, stride, reuse=False,
                  pad='SAME', dtype=tf.float32, bias=False, name='res_conv'):
    with tf.variable_scope(name):
        stride_shape = [1, stride, stride, 1]
        filter_shape = [filter_size, filter_size, input.get_shape()[3], num_filters]
        w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
        p = (filter_size - 1) // 2
        x = tf.pad(input, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        conv = tf.nn.conv2d(x, w, stride_shape, padding='VALID')
    return conv


def residual(input, num_filters, name, is_train, reuse=False, pad='REFLECT'):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse):
            out = residual_conv(input, num_filters, 3, 1, reuse, pad, name=name)
            out = tf.contrib.layers.batch_norm(
                out, center=True, scale=True, decay=0.9,
                is_training=is_train, updates_collections=None
            )
            out = tf.nn.relu(out)

        with tf.variable_scope('res2', reuse=reuse):
            out = residual_conv(out, num_filters, 3, 1, reuse, pad, name=name)
            out = tf.contrib.layers.batch_norm(
                out, center=True, scale=True, decay=0.9,
                is_training=is_train, updates_collections=None
            )

        return tf.nn.relu(input + out)


def fc(input, output_shape, is_train, info=False, batch_norm=True,
       activation_fn=lrelu, name="fc"):
    with tf.variable_scope(name):
        _ = slim.fully_connected(input, output_shape, activation_fn=None)
        _ = bn_act(_, is_train, batch_norm=batch_norm, activation_fn=activation_fn)
        if info: log.info('{} {}'.format(name, _))
    return _
