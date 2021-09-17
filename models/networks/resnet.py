#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-17-21 11:52
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import tensorflow as tf


class ResidualBlock(object):
    """ResNet basic block with weight norm."""

    def __init__(self, filters, weight_norm=True, name=None, is_training=True):

        self.filters = filters
        self.weight_norm = weight_norm
        self.name = name
        self.is_training = is_training

    def forward(self, x):
        with tf.variable_scope(self.name):
            skip = x
            x = tf.layers.batch_normalization(
                x, training=self.is_training, name='in_norm')
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=self.filters,
                                 kernel_size=3, padding='same', name='in_conv')

            x = tf.layers.batch_normalization(
                x, training=self.is_training, name='out_norm')
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=self.filters,
                                 kernel_size=3, padding='same', name='out_conv')
            x = x + skip

            return x


def run_layer_test():
    # Build graph first
    input_ = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_')
    resnet = ResidualBlock(filters=3, weight_norm=True,
                           name="residual_block_1", is_training=True)
    output = resnet.forward(input_)

    # Execute with a session
    with tf.Session() as sess:
        # Initialize variables
        # Same effect with below
        # tf.global_variables_initializer().run()
        sess.run(tf.global_variables_initializer())

        x_ = tf.random.uniform([64, 32, 32, 3], dtype=tf.float32)

        # {input_: x_}, 不要重名
        output = sess.run(output, feed_dict={input_: x_.eval()})

        print(type(output))
        print(output.shape)


def main():
    run_layer_test()


if __name__ == "__main__":
    main()
