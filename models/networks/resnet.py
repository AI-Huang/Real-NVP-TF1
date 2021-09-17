#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-17-21 11:52
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import tensorflow as tf


class ResidualBlock(object):
    """ResNet basic block with weight norm.

    TODO weight norm feature

    """

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


class ResNet(object):
    """ResNet for scale and translate factors in Real NVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        out_channels (int): Number of channels in the output.
        num_blocks (int): Number of residual blocks in the network.
        kernel_size (int): Side length of each filter in convolutional layers.
        padding (int): Padding for convolutional layers.
        double_after_norm (bool): Double input after input BatchNorm.
    """

    def __init__(self, mid_channels, out_channels,
                 num_blocks, kernel_size, padding, double_after_norm, name=None, is_training=True):

        # self.in_channels = in_channels # Not needed for TF1
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.padding = padding
        self.double_after_norm = double_after_norm
        self.name = name
        self.is_training = is_training

        self.blocks = [
            ResidualBlock(
                filters=mid_channels,
                weight_norm=True,
                name=f"residual_block_{i}", is_training=True)
            for i in range(num_blocks)]

    def forward(self, x):
        with tf.variable_scope(self.name):
            x = tf.layers.batch_normalization(
                x, training=self.is_training, name='in_norm')
            if self.double_after_norm:
                x *= 2.
            # TODO, Unknown trick
            # x = torch.cat((x, -x), dim=1)
            x = tf.concat((x, -x), axis=3)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=self.mid_channels,
                                 kernel_size=self.kernel_size, padding='same', name='in_conv')
            x_skip = tf.layers.conv2d(x, filters=self.mid_channels,
                                      kernel_size=1, padding='same', name='in_skip')

            for i, block in enumerate(self.blocks):
                x = block.forward(x)
                x_skip += tf.layers.conv2d(x,
                                           filters=self.mid_channels,
                                           kernel_size=1, padding='valid', name=f'skip_{i}')

            x = tf.layers.batch_normalization(
                x_skip, training=self.is_training, name='out_norm')
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(
                x_skip, filters=self.out_channels, kernel_size=1, padding='same', name='out_conv')

            return x
